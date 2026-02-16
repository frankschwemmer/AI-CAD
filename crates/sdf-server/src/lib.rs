use std::collections::BTreeMap;
use std::pin::Pin;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use http::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
use http::{Method, StatusCode};
use sdf_dsl::{Scene, compile_dsl};
use sdf_mesh::{Mesh, to_binary_stl};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

const DEFAULT_RESOLUTION: usize = 64;
const MAX_TRIANGLES: usize = 10_000_000;
const WS_DEBOUNCE_WINDOW: Duration = Duration::from_millis(50);

pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/evaluate", post(evaluate))
        .route("/evaluate/stl", post(evaluate_stl))
        .route("/modify", post(modify))
        .route("/validate", post(validate))
        .route("/ws", get(websocket))
        .layer(cors_layer())
}

fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any)
}

#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    dsl: String,
    resolution: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ModifyRequest {
    dsl: String,
    params: BTreeMap<String, f64>,
    resolution: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ValidateRequest {
    dsl: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WsClientMessage {
    SetDsl {
        dsl: String,
        resolution: Option<usize>,
    },
    SetParam {
        name: String,
        value: f64,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluateResponse {
    mesh: MeshJson,
    stats: EvaluateStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct MeshJson {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluateStats {
    time_ms: f64,
    triangle_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, Serialize, Deserialize)]
struct ValidateResponse {
    valid: bool,
    errors: Vec<String>,
    params: Vec<String>,
    bounding_box: BoundingBox,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct BoundingBox {
    min: [f64; 3],
    max: [f64; 3],
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WsServerMessage {
    Mesh {
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        time_ms: f64,
    },
    Error {
        message: String,
    },
}

#[derive(Debug)]
struct WsSessionState {
    scene: Scene,
    resolution: usize,
}

impl BoundingBox {
    fn empty() -> Self {
        Self {
            min: [0.0, 0.0, 0.0],
            max: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn evaluate(body: Bytes) -> Result<Json<EvaluateResponse>, ApiError> {
    let request: EvaluateRequest = parse_json(&body)?;
    let resolution = resolve_resolution(request.resolution)?;
    enforce_triangle_limit(resolution)?;

    let scene = compile_dsl(&request.dsl)
        .map_err(|err| ApiError::bad_request(format!("invalid DSL: {err}")))?;

    let (mesh, time_ms) = generate_mesh(&scene, resolution)?;
    Ok(Json(mesh_response(mesh, time_ms)))
}

async fn evaluate_stl(body: Bytes) -> Result<Response, ApiError> {
    let request: EvaluateRequest = parse_json(&body)?;
    let resolution = resolve_resolution(request.resolution)?;
    enforce_triangle_limit(resolution)?;

    let scene = compile_dsl(&request.dsl)
        .map_err(|err| ApiError::bad_request(format!("invalid DSL: {err}")))?;

    let (mesh, _) = generate_mesh(&scene, resolution)?;
    let bytes = to_binary_stl(&mesh, "mesh");

    let mut response = Response::new(axum::body::Body::from(bytes));
    response.headers_mut().insert(
        CONTENT_TYPE,
        "application/octet-stream"
            .parse()
            .expect("valid content type"),
    );
    response.headers_mut().insert(
        CONTENT_DISPOSITION,
        "attachment; filename=\"mesh.stl\""
            .parse()
            .expect("valid content disposition"),
    );
    Ok(response)
}

async fn modify(body: Bytes) -> Result<Json<EvaluateResponse>, ApiError> {
    let request: ModifyRequest = parse_json(&body)?;
    let resolution = resolve_resolution(request.resolution)?;
    enforce_triangle_limit(resolution)?;

    let mut scene = compile_dsl(&request.dsl)
        .map_err(|err| ApiError::bad_request(format!("invalid DSL: {err}")))?;

    for (name, value) in request.params {
        scene
            .set_param(&name, value)
            .map_err(|err| ApiError::bad_request(format!("parameter update failed: {err}")))?;
    }

    let (mesh, time_ms) = generate_mesh(&scene, resolution)?;
    Ok(Json(mesh_response(mesh, time_ms)))
}

async fn validate(body: Bytes) -> Json<ValidateResponse> {
    let request = match parse_json::<ValidateRequest>(&body) {
        Ok(request) => request,
        Err(err) => {
            return Json(ValidateResponse {
                valid: false,
                errors: vec![err.message],
                params: Vec::new(),
                bounding_box: BoundingBox::empty(),
            });
        }
    };

    match compile_dsl(&request.dsl) {
        Ok(scene) => {
            let (min, max) = scene.suggested_bounds();
            Json(ValidateResponse {
                valid: true,
                errors: Vec::new(),
                params: scene.parameter_names(),
                bounding_box: BoundingBox { min, max },
            })
        }
        Err(err) => Json(ValidateResponse {
            valid: false,
            errors: vec![err.to_string()],
            params: Vec::new(),
            bounding_box: BoundingBox::empty(),
        }),
    }
}

async fn websocket(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(mut socket: WebSocket) {
    let mut state: Option<WsSessionState> = None;
    let mut pending_params: BTreeMap<String, f64> = BTreeMap::new();
    let mut debounce_timer: Option<Pin<Box<tokio::time::Sleep>>> = None;

    loop {
        tokio::select! {
            _ = debounce_elapsed(&mut debounce_timer), if debounce_timer.is_some() => {
                if flush_param_updates(&mut socket, &mut state, &mut pending_params)
                    .await
                    .is_err()
                {
                    break;
                }
                debounce_timer = None;
            }
            message = socket.recv() => {
                let Some(message) = message else {
                    break;
                };

                match message {
                    Ok(Message::Text(text)) => {
                        if handle_ws_text_message(
                            &mut socket,
                            text.as_str(),
                            &mut state,
                            &mut pending_params,
                            &mut debounce_timer,
                        )
                        .await
                        .is_err()
                        {
                            break;
                        }
                    }
                    Ok(Message::Binary(_)) => {
                        if send_ws_error(&mut socket, "binary messages are not supported")
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(Message::Ping(payload)) => {
                        if socket.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Message::Pong(_)) => {}
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                }
            }
        }
    }
}

async fn debounce_elapsed(timer: &mut Option<Pin<Box<tokio::time::Sleep>>>) {
    if let Some(timer) = timer.as_mut() {
        timer.await;
    }
}

async fn handle_ws_text_message(
    socket: &mut WebSocket,
    raw_message: &str,
    state: &mut Option<WsSessionState>,
    pending_params: &mut BTreeMap<String, f64>,
    debounce_timer: &mut Option<Pin<Box<tokio::time::Sleep>>>,
) -> Result<(), ()> {
    let message: WsClientMessage = match serde_json::from_str(raw_message) {
        Ok(message) => message,
        Err(err) => {
            send_ws_error(socket, format!("invalid message: {err}")).await?;
            return Ok(());
        }
    };

    match message {
        WsClientMessage::SetDsl { dsl, resolution } => {
            pending_params.clear();
            *debounce_timer = None;
            handle_ws_set_dsl(socket, state, dsl, resolution).await
        }
        WsClientMessage::SetParam { name, value } => {
            pending_params.insert(name, value);
            *debounce_timer = Some(Box::pin(tokio::time::sleep(WS_DEBOUNCE_WINDOW)));
            Ok(())
        }
    }
}

async fn handle_ws_set_dsl(
    socket: &mut WebSocket,
    state: &mut Option<WsSessionState>,
    dsl: String,
    resolution: Option<usize>,
) -> Result<(), ()> {
    let resolution = match resolve_resolution(resolution) {
        Ok(value) => value,
        Err(err) => {
            send_ws_error(socket, err.message).await?;
            return Ok(());
        }
    };

    if let Err(err) = enforce_triangle_limit(resolution) {
        send_ws_error(socket, err.message).await?;
        return Ok(());
    }

    let scene = match compile_dsl(&dsl) {
        Ok(scene) => scene,
        Err(err) => {
            send_ws_error(socket, format!("invalid DSL: {err}")).await?;
            return Ok(());
        }
    };

    let (mesh, time_ms) = match generate_mesh(&scene, resolution) {
        Ok(result) => result,
        Err(err) => {
            send_ws_error(socket, err.message).await?;
            return Ok(());
        }
    };

    *state = Some(WsSessionState { scene, resolution });
    send_ws_mesh(socket, mesh, time_ms).await
}

async fn flush_param_updates(
    socket: &mut WebSocket,
    state: &mut Option<WsSessionState>,
    pending_params: &mut BTreeMap<String, f64>,
) -> Result<(), ()> {
    if pending_params.is_empty() {
        return Ok(());
    }

    let Some(state) = state.as_mut() else {
        pending_params.clear();
        send_ws_error(socket, "set_dsl must be called before set_param").await?;
        return Ok(());
    };

    for (name, value) in pending_params.iter() {
        if let Err(err) = state.scene.set_param(name, *value) {
            pending_params.clear();
            send_ws_error(socket, format!("parameter update failed: {err}")).await?;
            return Ok(());
        }
    }
    pending_params.clear();

    let (mesh, time_ms) = match generate_mesh(&state.scene, state.resolution) {
        Ok(result) => result,
        Err(err) => {
            send_ws_error(socket, err.message).await?;
            return Ok(());
        }
    };

    send_ws_mesh(socket, mesh, time_ms).await
}

async fn send_ws_mesh(socket: &mut WebSocket, mesh: Mesh, time_ms: f64) -> Result<(), ()> {
    send_ws_message(
        socket,
        WsServerMessage::Mesh {
            vertices: mesh.vertices,
            triangles: mesh.triangles,
            time_ms,
        },
    )
    .await
}

async fn send_ws_error(socket: &mut WebSocket, message: impl Into<String>) -> Result<(), ()> {
    send_ws_message(
        socket,
        WsServerMessage::Error {
            message: message.into(),
        },
    )
    .await
}

async fn send_ws_message(socket: &mut WebSocket, message: WsServerMessage) -> Result<(), ()> {
    let payload = serde_json::to_string(&message).map_err(|_| ())?;
    socket
        .send(Message::Text(payload.into()))
        .await
        .map_err(|_| ())
}

fn parse_json<T: DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
    if body.is_empty() {
        return Err(ApiError::bad_request("request body is required"));
    }

    serde_json::from_slice(body)
        .map_err(|err| ApiError::bad_request(format!("invalid JSON body: {err}")))
}

fn resolve_resolution(value: Option<usize>) -> Result<usize, ApiError> {
    let resolution = value.unwrap_or(DEFAULT_RESOLUTION);
    if resolution < 2 {
        return Err(ApiError::bad_request(
            "resolution must be at least 2 for marching cubes",
        ));
    }
    Ok(resolution)
}

fn enforce_triangle_limit(resolution: usize) -> Result<(), ApiError> {
    let cells_per_axis = resolution.saturating_sub(1);
    let max_triangles = 5usize
        .saturating_mul(cells_per_axis)
        .saturating_mul(cells_per_axis)
        .saturating_mul(cells_per_axis);

    if max_triangles > MAX_TRIANGLES {
        return Err(ApiError::payload_too_large(
            "requested resolution exceeds the 10M triangle safety limit",
        ));
    }

    Ok(())
}

fn generate_mesh(scene: &Scene, resolution: usize) -> Result<(Mesh, f64), ApiError> {
    let (min, max) = scene.suggested_bounds();
    let start = Instant::now();
    let mesh = scene
        .evaluate_mesh_with_bounds(resolution, min, max)
        .map_err(|err| ApiError::bad_request(format!("evaluation failed: {err}")))?;
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((mesh, time_ms))
}

fn mesh_response(mesh: Mesh, time_ms: f64) -> EvaluateResponse {
    EvaluateResponse {
        stats: EvaluateStats {
            time_ms,
            triangle_count: mesh.triangles.len(),
        },
        mesh: MeshJson {
            vertices: mesh.vertices,
            triangles: mesh.triangles,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::net::SocketAddr;
    use std::time::Duration;
    use std::time::Instant;

    use axum::Router;
    use axum::body::Body;
    use axum::response::Response;
    use futures::future::join_all;
    use futures::{SinkExt, StreamExt};
    use http::header::CONTENT_TYPE;
    use http::header::ORIGIN;
    use http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;
    use tokio::time::timeout;
    use tokio_tungstenite::tungstenite::Message as WsMessage;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
    use tower::ServiceExt;

    use super::{EvaluateResponse, MeshJson, ValidateResponse, app};

    #[tokio::test]
    async fn evaluate_returns_mesh_json() {
        let response = send_json(
            app(),
            Method::POST,
            "/evaluate",
            json!({"dsl": "sphere(10mm)", "resolution": 32}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let payload: EvaluateResponse = parse_json_response(response).await;
        assert!(!payload.mesh.vertices.is_empty());
        assert!(!payload.mesh.triangles.is_empty());
    }

    #[tokio::test]
    async fn evaluate_stl_returns_valid_binary_stl() {
        let response = send_json(
            app(),
            Method::POST,
            "/evaluate/stl",
            json!({"dsl": "sphere(10mm)", "resolution": 32}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = read_body_bytes(response).await;
        assert!(bytes.len() >= 84);

        let triangle_count =
            u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        assert_eq!(bytes.len(), 84 + triangle_count * 50);
    }

    #[tokio::test]
    async fn validate_reports_valid_program() {
        let response = send_json(
            app(),
            Method::POST,
            "/validate",
            json!({"dsl": "params { radius = 10mm } sphere(radius)"}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let payload: ValidateResponse = parse_json_response(response).await;
        assert!(payload.valid);
        assert!(payload.errors.is_empty());
        assert!(payload.params.contains(&"radius".to_string()));
    }

    #[tokio::test]
    async fn validate_reports_invalid_program() {
        let response = send_json(app(), Method::POST, "/validate", json!({"dsl": "sphere("})).await;

        assert_eq!(response.status(), StatusCode::OK);
        let payload: ValidateResponse = parse_json_response(response).await;
        assert!(!payload.valid);
        assert!(!payload.errors.is_empty());
    }

    #[tokio::test]
    async fn modify_changes_output_mesh() {
        let dsl = r#"
            params {
              radius = 10mm
            }
            sphere(radius)
        "#;

        let base_response = send_json(
            app(),
            Method::POST,
            "/evaluate",
            json!({"dsl": dsl, "resolution": 32}),
        )
        .await;
        assert_eq!(base_response.status(), StatusCode::OK);
        let base_payload: EvaluateResponse = parse_json_response(base_response).await;

        let modified_response = send_json(
            app(),
            Method::POST,
            "/modify",
            json!({
                "dsl": dsl,
                "params": { "radius": 20.0 },
                "resolution": 32
            }),
        )
        .await;
        assert_eq!(modified_response.status(), StatusCode::OK);
        let modified_payload: EvaluateResponse = parse_json_response(modified_response).await;

        let base_volume = mesh_volume(&base_payload.mesh).abs();
        let modified_volume = mesh_volume(&modified_payload.mesh).abs();
        assert!(modified_volume > base_volume * 2.0);
    }

    #[tokio::test]
    async fn evaluate_empty_body_returns_400() {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/evaluate")
            .body(Body::empty())
            .expect("request should build");

        let response = app()
            .oneshot(request)
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_value(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("request body")
        );
    }

    #[tokio::test]
    async fn evaluate_rejects_resolution_that_exceeds_triangle_limit() {
        let response = send_json(
            app(),
            Method::POST,
            "/evaluate",
            json!({"dsl": "sphere(10mm)", "resolution": 127}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = parse_json_value(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("10M triangle")
        );
    }

    #[tokio::test]
    async fn evaluate_round_trip_sphere_volume_is_reasonable() {
        let response = send_json(
            app(),
            Method::POST,
            "/evaluate",
            json!({"dsl": "sphere(10mm)", "resolution": 64}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let payload: EvaluateResponse = parse_json_response(response).await;
        let volume = mesh_volume(&payload.mesh).abs();
        let expected = (4.0 / 3.0) * PI * 10.0_f64.powi(3);
        let relative_error = ((volume - expected) / expected).abs();
        assert!(relative_error < 0.1);
    }

    #[tokio::test]
    async fn evaluate_endpoint_benchmark_meets_budget() {
        let start = Instant::now();
        let response = send_json(
            app(),
            Method::POST,
            "/evaluate",
            json!({"dsl": "sphere(10mm)", "resolution": 64}),
        )
        .await;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(response.status(), StatusCode::OK);
        let limit_ms = if cfg!(debug_assertions) {
            2500.0
        } else {
            500.0
        };
        assert!(
            elapsed_ms < limit_ms,
            "request exceeded budget: elapsed={elapsed_ms:.3}ms, limit={limit_ms:.1}ms"
        );
    }

    #[tokio::test]
    async fn concurrent_requests_complete_within_budget() {
        let app = app();
        let body = serde_json::to_vec(&json!({"dsl": "sphere(10mm)", "resolution": 64}))
            .expect("json encoding should succeed");

        let start = Instant::now();
        let futures = (0..10).map(|_| {
            let app = app.clone();
            let body = body.clone();
            async move {
                let request = Request::builder()
                    .method(Method::POST)
                    .uri("/evaluate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("request should build");
                app.oneshot(request).await.expect("request should complete")
            }
        });

        let responses = join_all(futures).await;
        let elapsed = start.elapsed().as_secs_f64();

        for response in responses {
            assert_eq!(response.status(), StatusCode::OK);
        }

        let limit_s = if cfg!(debug_assertions) { 15.0 } else { 5.0 };
        assert!(
            elapsed < limit_s,
            "concurrent budget exceeded: elapsed={elapsed:.3}s, limit={limit_s:.1}s"
        );
    }

    #[tokio::test]
    async fn cors_headers_are_present() {
        let request = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .header(ORIGIN, "https://example.com")
            .body(Body::empty())
            .expect("request should build");

        let response = app()
            .oneshot(request)
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        let allow_origin = response
            .headers()
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert_eq!(allow_origin, "*");
    }

    #[tokio::test]
    async fn websocket_set_dsl_returns_mesh_response() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };
        let mut socket = connect_ws(addr).await;

        send_ws_json(
            &mut socket,
            json!({"type": "set_dsl", "dsl": "sphere(10mm)", "resolution": 24}),
        )
        .await;
        let message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("mesh response should arrive");

        assert_eq!(message["type"], "mesh");
        assert!(
            message["vertices"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
        );
        assert!(
            message["triangles"]
                .as_array()
                .is_some_and(|t| !t.is_empty())
        );
        assert!(message["time_ms"].as_f64().unwrap_or_default() > 0.0);

        shutdown_ws_test(socket, server).await;
    }

    #[tokio::test]
    async fn websocket_set_param_returns_updated_mesh() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };
        let mut socket = connect_ws(addr).await;

        let dsl = r#"
            params {
              radius = 10mm
            }
            sphere(radius)
        "#;

        send_ws_json(
            &mut socket,
            json!({"type": "set_dsl", "dsl": dsl, "resolution": 32}),
        )
        .await;
        let initial_message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("initial mesh response should arrive");
        let initial_mesh = mesh_from_ws_message(&initial_message);
        let initial_volume = mesh_volume(&initial_mesh).abs();

        send_ws_json(
            &mut socket,
            json!({"type": "set_param", "name": "radius", "value": 16.0}),
        )
        .await;
        let updated_message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("updated mesh response should arrive");
        let updated_mesh = mesh_from_ws_message(&updated_message);
        let updated_volume = mesh_volume(&updated_mesh).abs();

        assert_eq!(updated_message["type"], "mesh");
        assert!(updated_volume > initial_volume * 2.0);

        shutdown_ws_test(socket, server).await;
    }

    #[tokio::test]
    async fn websocket_rapid_param_updates_are_debounced() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };
        let mut socket = connect_ws(addr).await;

        let dsl = r#"
            params {
              radius = 10mm
            }
            sphere(radius)
        "#;

        send_ws_json(
            &mut socket,
            json!({"type": "set_dsl", "dsl": dsl, "resolution": 24}),
        )
        .await;
        let _initial_message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("initial mesh response should arrive");

        for radius in 10..20 {
            send_ws_json(
                &mut socket,
                json!({"type": "set_param", "name": "radius", "value": radius as f64}),
            )
            .await;
        }

        let debounced = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("debounced mesh should arrive");
        assert_eq!(debounced["type"], "mesh");
        let max_radius = max_distance_from_origin(&mesh_from_ws_message(&debounced));
        assert!(
            max_radius > 18.0,
            "expected final radius close to 19mm, got {max_radius:.3}"
        );

        let extra = recv_ws_json(&mut socket, Duration::from_millis(250)).await;
        assert!(
            extra.is_none(),
            "expected a single debounced mesh update, got additional message: {extra:?}"
        );

        shutdown_ws_test(socket, server).await;
    }

    #[tokio::test]
    async fn websocket_invalid_dsl_returns_error_and_connection_stays_open() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };
        let mut socket = connect_ws(addr).await;

        send_ws_json(&mut socket, json!({"type": "set_dsl", "dsl": "sphere("})).await;
        let error_message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("error response should arrive");
        assert_eq!(error_message["type"], "error");
        assert!(
            error_message["message"]
                .as_str()
                .unwrap_or_default()
                .contains("invalid DSL")
        );

        send_ws_json(
            &mut socket,
            json!({"type": "set_dsl", "dsl": "sphere(6mm)", "resolution": 20}),
        )
        .await;
        let recovered_message = recv_ws_json(&mut socket, Duration::from_secs(3))
            .await
            .expect("valid mesh response should arrive");
        assert_eq!(recovered_message["type"], "mesh");

        shutdown_ws_test(socket, server).await;
    }

    #[tokio::test]
    async fn websocket_disconnect_and_reconnect_works() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };

        {
            let mut first_socket = connect_ws(addr).await;
            send_ws_json(
                &mut first_socket,
                json!({"type": "set_dsl", "dsl": "sphere(8mm)", "resolution": 20}),
            )
            .await;
            let first_message = recv_ws_json(&mut first_socket, Duration::from_secs(3))
                .await
                .expect("first connection should receive mesh");
            assert_eq!(first_message["type"], "mesh");
            first_socket
                .close(None)
                .await
                .expect("first socket should close");
        }

        {
            let mut second_socket = connect_ws(addr).await;
            send_ws_json(
                &mut second_socket,
                json!({"type": "set_dsl", "dsl": "sphere(8mm)", "resolution": 20}),
            )
            .await;
            let second_message = recv_ws_json(&mut second_socket, Duration::from_secs(3))
                .await
                .expect("second connection should receive mesh");
            assert_eq!(second_message["type"], "mesh");
            second_socket
                .close(None)
                .await
                .expect("second socket should close");
        }

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn websocket_set_param_latency_meets_budget() {
        let Some((addr, server)) = spawn_test_server().await else {
            return;
        };
        let mut socket = connect_ws(addr).await;

        let dsl = r#"
            params {
              radius = 10mm
            }
            sphere(radius)
        "#;

        send_ws_json(
            &mut socket,
            json!({"type": "set_dsl", "dsl": dsl, "resolution": 64}),
        )
        .await;
        let _initial = recv_ws_json(&mut socket, Duration::from_secs(4))
            .await
            .expect("initial mesh should arrive");

        let start = Instant::now();
        send_ws_json(
            &mut socket,
            json!({"type": "set_param", "name": "radius", "value": 12.0}),
        )
        .await;
        let message = recv_ws_json(&mut socket, Duration::from_secs(4))
            .await
            .expect("mesh update should arrive");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(message["type"], "mesh");
        let limit_ms = if cfg!(debug_assertions) {
            3000.0
        } else {
            200.0
        };
        assert!(
            elapsed_ms < limit_ms,
            "latency exceeded budget: elapsed={elapsed_ms:.3}ms, limit={limit_ms:.1}ms"
        );

        shutdown_ws_test(socket, server).await;
    }

    async fn send_json(
        router: Router,
        method: Method,
        uri: &str,
        value: serde_json::Value,
    ) -> Response {
        let body = serde_json::to_vec(&value).expect("json encoding should succeed");
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .expect("request should build");

        router
            .oneshot(request)
            .await
            .expect("request should complete")
    }

    async fn spawn_test_server() -> Option<(SocketAddr, JoinHandle<()>)> {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                eprintln!(
                    "skipping websocket test: local socket bind not permitted in this environment ({err})"
                );
                return None;
            }
            Err(err) => panic!("listener should bind: {err}"),
        };
        let addr = listener
            .local_addr()
            .expect("listener should expose address");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app())
                .await
                .expect("test server should run");
        });
        Some((addr, handle))
    }

    async fn connect_ws(
        addr: SocketAddr,
    ) -> WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>> {
        let url = format!("ws://{addr}/ws");
        let (socket, _response) = connect_async(&url)
            .await
            .expect("websocket client should connect");
        socket
    }

    async fn send_ws_json(
        socket: &mut WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        value: serde_json::Value,
    ) {
        socket
            .send(WsMessage::Text(value.to_string()))
            .await
            .expect("websocket send should succeed");
    }

    async fn recv_ws_json(
        socket: &mut WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        timeout_duration: Duration,
    ) -> Option<serde_json::Value> {
        loop {
            let message = timeout(timeout_duration, socket.next()).await.ok()??.ok()?;
            match message {
                WsMessage::Text(text) => return serde_json::from_str(text.as_ref()).ok(),
                WsMessage::Binary(bytes) => return serde_json::from_slice(&bytes).ok(),
                WsMessage::Ping(payload) => {
                    socket.send(WsMessage::Pong(payload)).await.ok()?;
                }
                WsMessage::Pong(_) => {}
                WsMessage::Close(_) => return None,
                WsMessage::Frame(_) => {}
            }
        }
    }

    async fn shutdown_ws_test(
        mut socket: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        server: JoinHandle<()>,
    ) {
        let _ = socket.close(None).await;
        server.abort();
        let _ = server.await;
    }

    fn mesh_from_ws_message(message: &serde_json::Value) -> MeshJson {
        serde_json::from_value(json!({
            "vertices": message["vertices"],
            "triangles": message["triangles"]
        }))
        .expect("mesh payload should decode")
    }

    fn max_distance_from_origin(mesh: &MeshJson) -> f64 {
        mesh.vertices
            .iter()
            .map(|vertex| {
                (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt()
            })
            .fold(0.0, f64::max)
    }

    async fn parse_json_response<T: serde::de::DeserializeOwned>(response: Response) -> T {
        let bytes = read_body_bytes(response).await;
        serde_json::from_slice(&bytes).expect("response should decode as JSON")
    }

    async fn parse_json_value(response: Response) -> serde_json::Value {
        let bytes = read_body_bytes(response).await;
        serde_json::from_slice(&bytes).expect("response should decode as JSON")
    }

    async fn read_body_bytes(response: Response) -> axum::body::Bytes {
        response
            .into_body()
            .collect()
            .await
            .expect("response body should collect")
            .to_bytes()
    }

    fn mesh_volume(mesh: &MeshJson) -> f64 {
        mesh.triangles
            .iter()
            .map(|tri| {
                let a = mesh.vertices[tri[0] as usize];
                let b = mesh.vertices[tri[1] as usize];
                let c = mesh.vertices[tri[2] as usize];
                dot(a, cross(b, c)) / 6.0
            })
            .sum()
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
