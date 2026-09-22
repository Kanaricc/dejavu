use std::{env, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use dejavu::{Classifier, ClassifyError, ClassifyRequest, ClassifyResponse, classify};
use serde_json::json;

struct ApiError(ClassifyError);

impl From<ClassifyError> for ApiError {
    fn from(error: ClassifyError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            ClassifyError::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            ClassifyError::Upstream(message) => (StatusCode::BAD_GATEWAY, message),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

async fn classify_api(
    State(classifier): State<Arc<Classifier>>,
    Json(request): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, ApiError> {
    Ok(Json(classify(&classifier, request).await?))
}

#[tokio::main]
async fn main() {
    let base_url =
        env::var("VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_owned());
    let timeout = env::var("VLLM_TIMEOUT")
        .ok()
        .map(|value| value.parse::<f64>().expect("VLLM_TIMEOUT must be a number"))
        .unwrap_or(60.0);
    let classifier = Arc::new(Classifier::new(
        base_url,
        env::var("VLLM_MODEL").expect("VLLM_MODEL must be set"),
        env::var("VLLM_API_KEY").unwrap_or_else(|_| "EMPTY".to_owned()),
        Duration::from_secs_f64(timeout),
    ));
    let address = env::var("API_BIND")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_owned())
        .parse::<SocketAddr>()
        .expect("API_BIND must be a socket address");
    let app = Router::new()
        .route("/classify", post(classify_api))
        .with_state(classifier);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("failed to bind API server");
    println!("listening on http://{address}");
    axum::serve(listener, app).await.expect("API server failed");
}
