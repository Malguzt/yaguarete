import os
import sys
import time
import socket
from math import floor
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import torch
import psutil
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Ensure src is in path for relative/absolute imports to work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infrastructure.transformers_engine.models_handler import ModelsHandler
from infrastructure.transformers_engine.model_catalog import ModelComplexity, ModelCatalog
from infrastructure.observability.metrics import NODE_NAME
from infrastructure.observability.hardware_metrics_collector import HardwareMetricsCollector
from application.router.router_service import RouterService
from infrastructure.repositories.router_stats_repository import RouterStatsRepository
from infrastructure.transformers_engine.embedding_engine import EmbeddingEngine
from application.router.quality_evaluator import QualityEvaluator
from application.router.phoenix_feedback_scheduler import PhoenixFeedbackScheduler
import uuid

app = FastAPI(title="Yaguarete LLM Proxy", version="1.0.0")


# --- Telemetry ---
def configure_cpu_threads() -> None:
    """
    Configure PyTorch CPU threading to better use host CPUs while avoiding
    oversubscription with BLAS/OpenMP pools.
    """
    logical_cores = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    default_intra_threads = max(1, floor(physical_cores * 0.8))
    default_interop_threads = max(1, min(8, floor(default_intra_threads / 4)))

    intra_threads = int(os.getenv("TORCH_NUM_THREADS", str(default_intra_threads)))
    interop_threads = int(os.getenv("TORCH_NUM_INTEROP_THREADS", str(default_interop_threads)))

    os.environ.setdefault("OMP_NUM_THREADS", str(intra_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(intra_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(intra_threads))

    torch.set_num_threads(max(1, intra_threads))
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        pass

    print(
        "[INFO] CPU thread tuning applied: "
        f"logical={logical_cores}, physical={physical_cores}, intra={torch.get_num_threads()}, "
        f"interop={interop_threads}, OMP={os.getenv('OMP_NUM_THREADS')}"
    )


def setup_telemetry() -> None:
    """Sets up OpenTelemetry tracing exporting to Arize Phoenix."""
    if os.getenv("OTEL_ENABLED", "1") not in ("1", "true", "yes", "on", "TRUE", "True"):
        print("[INFO] OpenTelemetry disabled by OTEL_ENABLED")
        return

    resource = Resource(attributes={"service.name": "yaguarete"})
    provider = TracerProvider(resource=resource)

    endpoint_base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")
    endpoint = f"{endpoint_base}/v1/traces"
    parsed = urlparse(endpoint_base)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as e:
        print(f"[WARNING] OpenTelemetry endpoint unavailable ({host}:{port}). Tracing disabled. error={e}")
        return

    schedule_delay_ms = max(100, int(os.getenv("OTEL_BSP_SCHEDULE_DELAY_MS", "500")))
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint),
        schedule_delay_millis=schedule_delay_ms,
    )

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    print(f"[INFO] OpenTelemetry tracing enabled on {endpoint} (batch_delay_ms={schedule_delay_ms})")


# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "yaguarete/auto"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


class FeedbackSignal(BaseModel):
    score: Optional[float] = None
    label: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    source: Optional[str] = "user"
    user: Optional[str] = None


class FeedbackRequest(FeedbackSignal):
    request_id: Optional[str] = None
    completion_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    request_id: str
    completion_id: str
    feedback_score: float
    feedback_label: str
    effectiveness_old: float
    effectiveness_new: float
    feedback_alpha: float


def _resolve_feedback_target(request_id: Optional[str], completion_id: Optional[str]) -> tuple[str, str]:
    raw_request_id = (request_id or "").strip()
    raw_completion_id = (completion_id or "").strip()

    if not raw_request_id and not raw_completion_id:
        raise HTTPException(
            status_code=400,
            detail="request_id or completion_id is required",
        )

    if raw_request_id:
        resolved_request_id = raw_request_id
        return resolved_request_id, f"chatcmpl-{resolved_request_id}"

    prefix = "chatcmpl-"
    if raw_completion_id.startswith(prefix):
        resolved_request_id = raw_completion_id[len(prefix):].strip()
    else:
        resolved_request_id = raw_completion_id

    if not resolved_request_id:
        raise HTTPException(status_code=400, detail="Invalid completion_id")
    return resolved_request_id, f"chatcmpl-{resolved_request_id}"


def _normalize_feedback_signal(score: Optional[float], label: Optional[str], rating: Optional[int]) -> tuple[float, str]:
    positive_labels = {"up", "thumbs_up", "like", "liked", "positive", "good"}
    negative_labels = {"down", "thumbs_down", "dislike", "negative", "bad"}
    neutral_labels = {"neutral", "meh", "mixed"}

    canonical_label = (label or "").strip().lower()

    if rating is not None:
        if rating not in (-1, 0, 1):
            raise HTTPException(status_code=400, detail="rating must be one of -1, 0, 1")
        score_value = float(rating)
        if not canonical_label:
            canonical_label = "thumbs_up" if rating > 0 else "thumbs_down" if rating < 0 else "neutral"
    elif score is not None:
        score_value = float(score)
    elif canonical_label in positive_labels:
        score_value = 1.0
    elif canonical_label in negative_labels:
        score_value = -1.0
    elif canonical_label in neutral_labels:
        score_value = 0.0
    else:
        raise HTTPException(status_code=400, detail="Provide at least one valid signal: score, rating or label")

    if score_value < -1.0 or score_value > 1.0:
        raise HTTPException(status_code=400, detail="score must be between -1 and 1")

    if canonical_label in positive_labels:
        canonical_label = "thumbs_up"
    elif canonical_label in negative_labels:
        canonical_label = "thumbs_down"
    elif canonical_label in neutral_labels:
        canonical_label = "neutral"
    else:
        canonical_label = "custom"

    return score_value, canonical_label


def _apply_feedback_to_request(
    request_id: str,
    completion_id: str,
    payload: FeedbackSignal,
) -> FeedbackResponse:
    if not stats_repo.request_exists(request_id):
        raise HTTPException(status_code=404, detail=f"request_id not found: {request_id}")

    feedback_score, feedback_label = _normalize_feedback_signal(payload.score, payload.label, payload.rating)
    feedback_comment = (payload.comment or "").strip()[:2000]
    feedback_source = (payload.source or "user").strip().lower()[:64] or "user"
    feedback_user = (payload.user or "").strip()[:128]

    updated = stats_repo.apply_user_feedback(
        request_id=request_id,
        feedback_score=feedback_score,
        feedback_label=feedback_label,
        feedback_comment=feedback_comment,
        feedback_source=feedback_source,
        feedback_user=feedback_user,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"request_id not found: {request_id}")

    alpha = float(updated.get("feedback_alpha", os.getenv("ROUTER_FEEDBACK_ALPHA", "0.35")))
    alpha = max(0.01, min(alpha, 0.95))
    return FeedbackResponse(
        status="accepted",
        request_id=request_id,
        completion_id=completion_id,
        feedback_score=feedback_score,
        feedback_label=feedback_label,
        effectiveness_old=updated["old_effectiveness"],
        effectiveness_new=updated["new_effectiveness"],
        feedback_alpha=alpha,
    )


# --- State ---
models_handler = ModelsHandler()
metrics_collector = HardwareMetricsCollector(interval=5)
stats_repo = RouterStatsRepository()
embedding_engine = EmbeddingEngine()
router_service = RouterService(stats_repo, embedding_engine)
quality_evaluator = QualityEvaluator(models_handler)
phoenix_feedback_scheduler = PhoenixFeedbackScheduler(stats_repo=stats_repo, quality_evaluator=quality_evaluator)
tracer = trace.get_tracer("yaguarete.router")


@app.on_event("startup")
async def startup_event():
    print(f"[INFO] Yaguarete starting on node: {NODE_NAME}")
    configure_cpu_threads()
    setup_telemetry()
    metrics_collector.start()
    models_handler.preload_models()
    phoenix_feedback_scheduler.start()
    print("[INFO] Model preload started in background...")


@app.on_event("shutdown")
async def shutdown_event():
    phoenix_feedback_scheduler.stop()
    metrics_collector.stop()


# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "node": NODE_NAME}


@app.get("/v1/models")
async def list_models():
    catalog = ModelCatalog()
    available_models = []
    for model_def in catalog.models:
        available_models.append(
            {
                "id": model_def.huggingface_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "yaguarete" if not model_def.is_remote else model_def.provider.value,
                "complexity": model_def.complexity.value,
                "specialty": model_def.specialty.value,
                "is_remote": model_def.is_remote,
                "supports_generation": model_def.supports_generation,
                "provider": model_def.provider.value,
                "cost_per_1k_chars": model_def.cost_per_1k_chars,
                "estimated_vram_gb": model_def.estimated_vram_gb,
            }
        )
    return {"object": "list", "data": available_models}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")

    prompt = user_messages[-1]
    session_id = request.user or "default-session"
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        with tracer.start_as_current_span("yaguarete.chat_completion") as span:
            span.set_attribute("yaguarete.request_id", request_id)
            span.set_attribute("yaguarete.session_id", session_id)
            span.set_attribute("yaguarete.input", prompt[:8000])

            embedding = embedding_engine.get_embedding(prompt)
            catalog = ModelCatalog()

            if not request.model or request.model == "yaguarete/auto":
                model_id = router_service.route_request(prompt, session_id, embedding)
            else:
                model_id = request.model
                explicit_model = catalog.get_model(model_id)
                if explicit_model is None:
                    raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
                if not explicit_model.supports_generation:
                    raise HTTPException(status_code=400, detail=f"Model does not support generation: {model_id}")

            print(f"[INFO] Routing to model: {model_id}")
            span.set_attribute("yaguarete.model_id", model_id)

            required_complexity = None
            for model_def in catalog.models:
                if model_def.huggingface_id == model_id:
                    required_complexity = model_def.complexity
                    break

            try:
                response_text = models_handler.generate_text(
                    prompt,
                    required_complexity=required_complexity,
                    model_id=model_id,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature if request.temperature is not None else 0.7,
                )
            except Exception as first_error:
                fallback_model = catalog.get_default_model()
                if fallback_model.huggingface_id == model_id:
                    raise first_error
                print(
                    f"[WARNING] Primary model {model_id} failed. Retrying with local fallback "
                    f"{fallback_model.huggingface_id}. error={first_error}"
                )
                model_id = fallback_model.huggingface_id
                span.set_attribute("yaguarete.model_id", model_id)
                required_complexity = fallback_model.complexity
                response_text = models_handler.generate_text(
                    prompt,
                    required_complexity=required_complexity,
                    model_id=model_id,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature if request.temperature is not None else 0.7,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("yaguarete.duration_ms", duration_ms)
            span.set_attribute("yaguarete.output", response_text[:8000])

            cost = 0.0
            for m in catalog.models:
                if m.huggingface_id == model_id:
                    cost = (len(prompt) + len(response_text)) * (m.cost_per_1k_chars / 1000)
                    break

            stats_repo.log_request(
                {
                    "model_id": model_id,
                    "request_id": request_id,
                    "input_chars": len(prompt),
                    "output_chars": len(response_text),
                    "duration_ms": duration_ms,
                    "cost": cost,
                    "topic": "general",
                    "session_id": session_id,
                    "embedding": embedding,
                    "format_score": 1.0,
                    "density_score": 1.0,
                    "judge_score": 1.0,
                    "sentiment_score": 0.0,
                }
            )

            # Deferred quality process: starts after inactivity window.
            phoenix_feedback_scheduler.register_completed_request(
                request_id=request_id,
                model_id=model_id,
                duration_ms=duration_ms,
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            model=model_id,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=response_text),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    request_id, completion_id = _resolve_feedback_target(request.request_id, request.completion_id)
    return _apply_feedback_to_request(request_id, completion_id, request)


@app.post("/v1/chat/completions/{completion_id}/feedback", response_model=FeedbackResponse)
async def submit_completion_feedback(completion_id: str, request: FeedbackSignal):
    request_id, resolved_completion_id = _resolve_feedback_target(None, completion_id)
    return _apply_feedback_to_request(request_id, resolved_completion_id, request)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
