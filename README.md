# dejavu

**Run a jev-like model without any training.**

dejavu demonstrates how to turn any vLLM service into a JEV-like classification API without training the model.

The reported `candidate_probabilities` can be useful as confidence-like scores, but this project does not *claim* that they are calibrated probabilities or that they carry statistical interpretation. Empirically, however, the resulting evaluation metrics are strong.

## Evaluation

> https://github.com/fstandhartinger/jevbench
>
> only open-source 231 problems.

| Rank | Model | Acc | P50 |
| --- | --- | --- | --- |
| 1 | DeepSeek V4.1 Flash | 97.84% | 1.501s |
| 2 | GPT-5.6 Luna | 97.40% | 0.975s |
| 3 | OpenJev thinking | 88.74% | 0.512s |
| 4 | djev thinking | 87.45% | 0.433s |
| 5= | Gemini 3.1 Flash-Lite | 87.01% | 0.768s |
| 5= | reflex-27b | 87.01% | 1.892s |
| 7= | **dejavu** | **86.58%** | **~ 0.7s** |
| 7= | Jev 1.13.0 | 86.58% | 0.665s |
| 7= | Simple.Jev Qwen3.8-27B | 86.58% | 1.107s |
| 10 | Lit.Jev Qwen3.8-27B | 86.15% | 1.917s |

## Quickstart

### 1. Start vLLM

dejavu expects an OpenAI-compatible vLLM endpoint with reasoning output, structured outputs, and token log probabilities enabled. For a Qwen3 model, start vLLM with its reasoning parser:

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

### 2. Start the API

```bash
VLLM_MODEL=Qwen/Qwen3-8B \
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
cargo run --release --bin dejavu-api
```

The API listens on `127.0.0.1:3000` by default. Set `API_BIND`, `VLLM_API_KEY`, or `VLLM_TIMEOUT` to override the corresponding defaults.

### 3. Classify

```bash
curl http://127.0.0.1:3000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "subject: Refund request\nbody: I was charged twice",
    "question": "Which department should handle this?",
    "options": [
      "payments, invoices and refunds",
      "bugs and outages",
      "pricing and new contracts"
    ]
  }'
```

The response contains the selected answer, per-candidate log probabilities and normalized probabilities, the model used, and the number of vLLM requests made during classification.
