# Yaguarete 🐆

LLM Model Proxy and Routing Web Service.

Yaguarete acts as a guide (guía) for selecting and serving LLMs efficiently, with:
- OpenRouter/OpenAI compatible API.
- Intelligent model routing based on hardware profile.
- VRAM optimization and memory management.
- Real-time hardware monitoring (GPU/CPU/RAM).
- Observability via Arize Phoenix and Prometheus.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python src/main.py
```

## API

- `POST /v1/chat/completions`: Generate text (OpenAI compatible).
- `GET /v1/models`: List available models.
- `GET /metrics`: Prometheus metrics.
