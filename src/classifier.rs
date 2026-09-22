use std::{collections::HashMap, future::Future, pin::Pin, time::Duration};

use reqwest::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct Classifier {
    client: Client,
    base_url: String,
    model: String,
    api_key: String,
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_state")]
    state: String,
    question: String,
    options: Vec<String>,
    #[serde(default = "default_thinking_budget")]
    thinking_budget: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_top_logprobs")]
    top_logprobs: usize,
    #[serde(default = "default_mass_tolerance")]
    mass_tolerance: f64,
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    model: String,
    answer: String,
    candidate_logprobs: HashMap<String, Option<f64>>,
    candidate_probabilities: HashMap<String, f64>,
    probability_scope: &'static str,
    shared_prefix_removed: String,
    thinking_budget: usize,
    vllm_requests: usize,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<Message>,
    #[serde(default)]
    logprobs: Option<Logprobs>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct Logprobs {
    #[serde(default)]
    content: Vec<ContentLogprob>,
}

#[derive(Deserialize)]
struct ContentLogprob {
    #[serde(default)]
    top_logprobs: Vec<TokenLogprob>,
}

#[derive(Clone, Deserialize)]
struct TokenLogprob {
    token: String,
    logprob: f64,
}

pub enum ClassifyError {
    BadRequest(String),
    Upstream(String),
}

type AppState = Classifier;
type AppError = ClassifyError;

fn default_state() -> String {
    "subject: Refund request\nbody: I was charged twice".to_owned()
}

fn default_thinking_budget() -> usize {
    64
}

fn default_max_tokens() -> usize {
    4
}

fn default_top_logprobs() -> usize {
    20
}

fn default_mass_tolerance() -> f64 {
    1e-5
}

impl ClassifyRequest {
    fn validate(&self) -> Result<(), AppError> {
        if self.options.len() < 2 {
            return Err(AppError::BadRequest(
                "at least two options are required".to_owned(),
            ));
        }
        if self.options.iter().any(|option| option.trim().is_empty()) {
            return Err(AppError::BadRequest(
                "option text cannot be empty".to_owned(),
            ));
        }
        for (index, option) in self.options.iter().enumerate() {
            if self.options[..index].contains(option) {
                return Err(AppError::BadRequest(format!(
                    "duplicate option: {option:?}"
                )));
            }
        }
        if self.thinking_budget < 1 {
            return Err(AppError::BadRequest(
                "thinking_budget must be at least 1".to_owned(),
            ));
        }
        if self.max_tokens < 1 {
            return Err(AppError::BadRequest(
                "max_tokens must be at least 1".to_owned(),
            ));
        }
        if self.top_logprobs < 1 {
            return Err(AppError::BadRequest(
                "top_logprobs must be at least 1".to_owned(),
            ));
        }
        if !(0.0..1.0).contains(&self.mass_tolerance) {
            return Err(AppError::BadRequest(
                "mass_tolerance must be in [0, 1)".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Classifier {
    pub fn new(base_url: String, model: String, api_key: String, timeout: Duration) -> Self {
        Self {
            client: Client::builder()
                .timeout(timeout)
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_owned(),
            model,
            api_key,
        }
    }

    async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        payload: Value,
    ) -> Result<T, AppError> {
        let mut request = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(&payload);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }
        let response = request
            .send()
            .await
            .map_err(|error| AppError::Upstream(format!("cannot reach vLLM: {error}")))?;
        self.decode_response(response).await
    }

    async fn decode_response<T: DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> Result<T, AppError> {
        let status = response.status();
        if !status.is_success() {
            let detail = response
                .text()
                .await
                .unwrap_or_else(|error| error.to_string());
            return Err(AppError::Upstream(format!(
                "vLLM returned HTTP {}: {detail}",
                status.as_u16()
            )));
        }
        response
            .json()
            .await
            .map_err(|error| AppError::Upstream(format!("invalid vLLM response: {error}")))
    }
}

fn build_prompt(
    state: &str,
    question: &str,
    options: &[String],
    reasoning: Option<&str>,
) -> String {
    let question_data = json!({
        "type": "choice",
        "instructions": question,
        "options": options,
    });
    let mut parts =
        vec!["Select exactly one value from question.options. Return only that value.".to_owned()];
    parts.push(format!("state = {state}"));
    if let Some(reasoning) = reasoning {
        parts.push(format!(
            "Prior analysis to use as context without repeating it: {}",
            serde_json::to_string(reasoning).expect("a string is always JSON serializable")
        ));
    }
    parts.push(format!("question = {question_data}"));
    parts.join("\n\n")
}

fn remove_shared_prefix(options: &[String]) -> Result<(String, HashMap<String, String>), AppError> {
    for (index, option) in options.iter().enumerate() {
        for other in &options[index + 1..] {
            let (shorter, longer) = if option.len() <= other.len() {
                (option, other)
            } else {
                (other, option)
            };
            if longer.starts_with(shorter) {
                return Err(AppError::BadRequest(format!(
                    "option {shorter:?} is a prefix of {longer:?}; the options cannot be distinguished without an EOS probability"
                )));
            }
        }
    }

    let mut prefix = options[0].clone();
    for option in &options[1..] {
        prefix = prefix
            .chars()
            .zip(option.chars())
            .take_while(|(left, right)| left == right)
            .map(|(character, _)| character)
            .collect();
        if prefix.is_empty() {
            break;
        }
    }
    let remaining = options
        .iter()
        .map(|option| {
            (
                option.clone(),
                option
                    .strip_prefix(&prefix)
                    .expect("prefix was derived from every option")
                    .to_owned(),
            )
        })
        .collect();
    Ok((prefix, remaining))
}

async fn request_reasoning(
    state: &AppState,
    model: &str,
    prompt: &str,
    thinking_budget: usize,
) -> Result<String, AppError> {
    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Analyze the classification briefly. After thinking, return exactly one value from question.options without explanation."
            },
            {"role": "user", "content": prompt}
        ],
        "chat_template_kwargs": {
            "enable_thinking": true,
            "preserve_thinking": false
        },
        "reasoning_effort": "low",
        "thinking_token_budget": thinking_budget,
        "max_tokens": thinking_budget + 16,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20
    });
    let response: ChatResponse = state.post_json("/chat/completions", payload).await?;
    let message = response
        .choices
        .first()
        .and_then(|choice| choice.message.as_ref())
        .ok_or_else(|| AppError::Upstream("vLLM did not return a message".to_owned()))?;
    let reasoning = message
        .reasoning
        .as_deref()
        .or(message.reasoning_content.as_deref())
        .map(str::trim)
        .filter(|reasoning| !reasoning.is_empty())
        .ok_or_else(|| {
            AppError::Upstream(
                "vLLM did not return reasoning content; start it with --reasoning-parser qwen3"
                    .to_owned(),
            )
        })?;
    Ok(reasoning.to_owned())
}

async fn request_next_token(
    state: &AppState,
    request: &ClassifyRequest,
    model: &str,
    prompt: &str,
    generated_prefix: &str,
    allowed_suffixes: Vec<String>,
) -> Result<Vec<TokenLogprob>, AppError> {
    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a deterministic classifier. Do not explain or reason out loud. Complete the answer with exactly one allowed key."
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": format!("choice:{generated_prefix}")}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true,
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": true,
        "top_logprobs": request.top_logprobs,
        "structured_outputs": {"choice": allowed_suffixes}
    });
    let response: ChatResponse = state.post_json("/chat/completions", payload).await?;
    let alternatives = response
        .choices
        .first()
        .and_then(|choice| choice.logprobs.as_ref())
        .and_then(|logprobs| logprobs.content.first())
        .map(|content| &content.top_logprobs)
        .ok_or_else(|| AppError::Upstream("vLLM did not return token logprobs".to_owned()))?;
    if alternatives.is_empty() {
        return Err(AppError::Upstream(
            "vLLM returned an empty top_logprobs list".to_owned(),
        ));
    }
    Ok(alternatives.clone())
}

fn score_candidate_prefixes<'a>(
    state: &'a AppState,
    request: &'a ClassifyRequest,
    model: &'a str,
    prompt: &'a str,
    remaining: HashMap<String, String>,
    generated_prefix: String,
    path_probability: f64,
    depth: usize,
) -> Pin<Box<dyn Future<Output = Result<(HashMap<String, f64>, usize), AppError>> + Send + 'a>> {
    Box::pin(async move {
        let alternatives = request_next_token(
            state,
            request,
            model,
            prompt,
            &generated_prefix,
            remaining.values().cloned().collect(),
        )
        .await?;
        let mut token_probabilities: HashMap<String, f64> = HashMap::new();
        for item in alternatives {
            *token_probabilities.entry(item.token).or_default() += item.logprob.exp();
        }
        let returned_mass: f64 = token_probabilities.values().sum();
        if returned_mass < 1.0 - request.mass_tolerance {
            return Err(AppError::Upstream(format!(
                "top_logprobs covered only {returned_mass:.6} probability mass at prefix {generated_prefix:?}; increase top_logprobs"
            )));
        }

        let mut scores = remaining
            .keys()
            .map(|label| (label.clone(), 0.0))
            .collect::<HashMap<_, _>>();
        let mut request_count = 1;
        for (token, token_probability) in token_probabilities {
            if token.is_empty() {
                continue;
            }
            let matches = remaining
                .iter()
                .filter_map(|(label, suffix)| {
                    suffix
                        .strip_prefix(&token)
                        .map(|rest| (label.clone(), rest.to_owned()))
                })
                .collect::<HashMap<_, _>>();
            if matches.is_empty() {
                continue;
            }

            let branch_probability = path_probability * token_probability;
            if matches.len() == 1 {
                let label = matches.keys().next().expect("one match exists");
                *scores.get_mut(label).expect("score exists for every label") += branch_probability;
                continue;
            }
            if depth + 1 >= request.max_tokens {
                return Err(AppError::Upstream(format!(
                    "prefix {:?} is still ambiguous after {} tokens; increase max_tokens",
                    generated_prefix.clone() + &token,
                    request.max_tokens
                )));
            }

            let (branch_scores, branch_requests) = score_candidate_prefixes(
                state,
                request,
                model,
                prompt,
                matches,
                generated_prefix.clone() + &token,
                branch_probability,
                depth + 1,
            )
            .await?;
            request_count += branch_requests;
            for (label, probability) in branch_scores {
                *scores.get_mut(&label).expect("branch label exists") += probability;
            }
        }
        Ok((scores, request_count))
    })
}

pub async fn classify(
    state: &Classifier,
    mut request: ClassifyRequest,
) -> Result<ClassifyResponse, ClassifyError> {
    request.options = request
        .options
        .into_iter()
        .map(|option| option.trim().to_owned())
        .collect();
    request.validate()?;
    let options = request.options.clone();
    let (shared_prefix, remaining) = remove_shared_prefix(&options)?;
    let model = request.model.clone().unwrap_or_else(|| state.model.clone());
    let initial_prompt = build_prompt(&request.state, &request.question, &options, None);
    let reasoning =
        request_reasoning(&state, &model, &initial_prompt, request.thinking_budget).await?;
    let scoring_prompt = build_prompt(
        &request.state,
        &request.question,
        &options,
        Some(&reasoning),
    );
    let (raw_probabilities, request_count) = score_candidate_prefixes(
        &state,
        &request,
        &model,
        &scoring_prompt,
        remaining,
        format!(" {shared_prefix}"),
        1.0,
        0,
    )
    .await?;
    let total: f64 = raw_probabilities.values().sum();
    if total == 0.0 {
        return Err(AppError::Upstream(
            "no returned token prefix matched any option".to_owned(),
        ));
    }
    let probabilities = raw_probabilities
        .into_iter()
        .map(|(label, probability)| (label, probability / total))
        .collect::<HashMap<_, _>>();
    let candidate_logprobs = probabilities
        .iter()
        .map(|(label, probability)| {
            (
                label.clone(),
                (*probability > 0.0).then(|| probability.ln()),
            )
        })
        .collect();
    let winner = probabilities
        .iter()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(label, _)| label)
        .expect("at least two probabilities exist");

    Ok(ClassifyResponse {
        model,
        answer: format!("choice: {winner}"),
        candidate_logprobs,
        candidate_probabilities: probabilities,
        probability_scope: "normalized across the listed options",
        shared_prefix_removed: shared_prefix,
        thinking_budget: request.thinking_budget,
        vllm_requests: request_count + 1,
    })
}
