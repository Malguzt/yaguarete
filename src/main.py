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

# --- State ---

models_handler = ModelsHandler()
metrics_collector = HardwareMetricsCollector(interval=5)

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
    
    try:
        # Determine complexity from model-id if possible
        required_complexity = None
        for model_def in ModelCatalog().models:
            if model_def.huggingface_id == request.model:
                required_complexity = model_def.complexity
                break
        
        response_text = models_handler.generate_text(prompt, required_complexity=required_complexity)
        
        return ChatCompletionResponse(
            model=request.model,
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
