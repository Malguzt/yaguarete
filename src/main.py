import os
import sys
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import start_http_server, generate_latest, CONTENT_TYPE_LATEST
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
import uuid

app = FastAPI(title="Yaguarete LLM Proxy", version="1.0.0")

# --- Telemetry ---

def setup_telemetry():
    """Sets up OpenTelemetry tracing exporting to Arize Phoenix."""
    resource = Resource(attributes={
        "service.name": "yaguarete"
    })

    provider = TracerProvider(resource=resource)

    # Phoenix OTLP HTTP receiver is on port 4318, trace endpoint is /v1/traces
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    print("[INFO] OpenTelemetry tracing to Phoenix enabled on localhost:4318")

# --- Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
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
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

from application.router.quality_evaluator import QualityEvaluator

# --- State ---

models_handler = ModelsHandler()
metrics_collector = HardwareMetricsCollector(interval=5)
stats_repo = RouterStatsRepository()
embedding_engine = EmbeddingEngine()
router_service = RouterService(stats_repo, embedding_engine)
quality_evaluator = QualityEvaluator(models_handler)

@app.on_event("startup")
async def startup_event():
    print(f"[INFO] Yaguarete starting on node: {NODE_NAME}")
    setup_telemetry()
    metrics_collector.start()
    models_handler.preload_models()
    print("[INFO] Model preload started in background...")

@app.on_event("shutdown")
async def shutdown_event():
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
        available_models.append({
            "id": model_def.huggingface_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "yaguarete",
            "complexity": model_def.complexity.value,
            "specialty": model_def.specialty.value
        })
    return {"object": "list", "data": available_models}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # Route logic: extract the last user message as prompt
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")
    
    prompt = user_messages[-1]
    # Use OpenRouter 'user' field as session_id if available, or generate one
    session_id = getattr(request, 'user', None) or "default-session"
    
    start_time = time.perf_counter()
    
    try:
        # 1. Generate embedding first (Mandatory for similarity-based routing)
        embedding = embedding_engine.get_embedding(prompt)
        
        # 2. Determine model using Router Service
        if not request.model or request.model == "yaguarete/auto":
            model_id = router_service.route_request(prompt, session_id, embedding)
        else:
            model_id = request.model
            
        print(f"[INFO] Routing to model: {model_id}")
        
        # 3. Determine complexity for models_handler
        required_complexity = None
        for model_def in ModelCatalog().models:
            if model_def.huggingface_id == model_id:
                required_complexity = model_def.complexity
                break
        
        response_text = models_handler.generate_text(prompt, required_complexity=required_complexity)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate cost
        catalog = ModelCatalog()
        cost = 0.0
        for m in catalog.models:
            if m.huggingface_id == model_id:
                cost = (len(prompt) + len(response_text)) * (m.cost_per_1k_chars / 1000)
                break

        # Evaluate quality
        quality_scores = quality_evaluator.evaluate_response(prompt, response_text)
        
        # Shadowing: 5% chance to compare with another model
        shadow_model_id = router_service.select_shadow_model(model_id, required_complexity)
        if shadow_model_id:
            print(f"[SHADOW] Running comparison with {shadow_model_id}")
            try:
                shadow_resp = models_handler.generate_text(prompt, model_id=shadow_model_id)
                # Compare semantic similarity
                primary_emb = embedding_engine.get_embeddings(response_text)
                shadow_emb = embedding_engine.get_embeddings(shadow_resp)
                similarity = embedding_engine.calculate_similarity(primary_emb, shadow_emb)
                
                # If they diverge too much, slightly penalize both or flag for review
                if similarity < 0.7:
                    print(f"[SHADOW] High divergence detected! Similarity: {similarity:.2f}")
                    # Update judge_score to reflect uncertainty
                    quality_scores["judge_score"] *= 0.8
            except Exception as e:
                print(f"[SHADOW] Error during shadowing: {e}")

        # Log stats
        stats_repo.log_request({
            "model_id": model_id,
            "request_id": str(uuid.uuid4()),
            "input_chars": len(prompt),
            "output_chars": len(response_text),
            "duration_ms": duration_ms,
            "cost": cost,
            "topic": "general",
            "session_id": session_id,
            "embedding": embedding,
            **quality_scores
        })
        
        # Update Prometheus metrics
        from infrastructure.observability.metrics import ROUTER_MODEL_EFFECTIVENESS, ROUTER_AVG_TIME_PER_CHAR
        
        # Calculate a combined effectiveness for live monitoring
        combined_eff = (quality_scores["judge_score"] * 0.5) + (quality_scores["format_score"] * 0.3) + (quality_scores["density_score"] * 0.2)
        ROUTER_MODEL_EFFECTIVENESS.labels(model_id=model_id).set(combined_eff)
        ROUTER_AVG_TIME_PER_CHAR.labels(model_id=model_id).set(duration_ms / max(len(prompt), 1))
        
        return ChatCompletionResponse(
            model=model_id,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=response_text),
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),  # Crude estimation
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Start metrics server on a different port or handle same port
    # Prometheus client start_http_server(8000) was in donMingo
    # Here we expose /metrics via FastAPI, but we can also start the standard server
    # Let's use FastAPI's endpoint for consistency if preferred, 
    # but some Prometheus scrapers prefer a dedicated port.
    uvicorn.run(app, host="0.0.0.0", port=8001)
