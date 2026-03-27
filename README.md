# Yaguarete 🐆

Proxy de Modelos LLM y Servicio de Enrutamiento Inteligente.

Yaguarete actúa como un guía para la selección y servicio eficiente de LLMs, optimizando el uso de hardware local y reduciendo costos de inferencia.

## Características Principales

- **API Compatible**: Soporta los estándares de OpenAI y OpenRouter (`/v1/chat/completions`).
- **Enrutamiento k-NN**: Selección dinámica de modelos basada en la **similitud semántica** de pedidos previos (proximidad vectorial).
- **Post-Evaluación Semántica**: Evaluación automática de respuestas locally (Juez LLM, Formato, Densidad y Sentimiento).
- **Monitoreo de Hardware**: Telemetría en tiempo real de GPU (vRAM), CPU, RAM e I/O.
- **Observabilidad**: Integración nativa con **Arize Phoenix** para trazas y **Prometheus/Grafana** para métricas de performance.
- **Optimización de VRAM**: Gestión centralizada de modelos con carga/descarga dinámica.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python src/main.py
```

## Endpoints

- `POST /v1/chat/completions`: Generación de texto (compatible con OpenAI).
- `GET /v1/models`: Listado de modelos disponibles en el catálogo.
- `GET /metrics`: Métricas para Prometheus.

---
Para más detalles sobre el algoritmo de enrutamiento, consulta [router_documentation.md](file:///home/jose/.gemini/antigravity/brain/c6e11bcb-8590-4413-8313-385a5a46575e/router_documentation.md).
