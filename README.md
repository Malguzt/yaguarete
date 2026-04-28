# Yaguarete 🐆

Proxy de Modelos LLM y Servicio de Enrutamiento Inteligente.

Yaguarete actúa como un guía para la selección y servicio eficiente de LLMs, optimizando el uso de hardware local y reduciendo costos de inferencia.

## Características Principales

- **API Compatible**: Soporta los estándares de OpenAI y OpenRouter (`/v1/chat/completions`).
- **Enrutamiento k-NN**: Selección dinámica de modelos basada en la **similitud semántica** de pedidos previos (proximidad vectorial).
- **Modo Local-First**: Por defecto prioriza modelos locales para evitar costos y depender menos de endpoints externos.
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

## Ajuste de Modelos con llmfit

Si estás corriendo `llmfit` como servicio (`llmfit serve`), Yaguarete puede usar ese ranking para priorizar qué modelos locales intenta primero.

Variables opcionales:

```bash
LLMFIT_SERVICE_URL=http://127.0.0.1:8787
LLMFIT_MIN_FIT=good
LLMFIT_LIMIT=12
LLMFIT_REPLACE_LOCAL_MODELS=true
LLMFIT_CACHE_TTL_SEC=900
LLMFIT_SNAPSHOT_PATH=data/llmfit_local_catalog_snapshot.json
LLMFIT_VALIDATE_MODEL_LOADABILITY=1
```

Si `LLMFIT_REPLACE_LOCAL_MODELS=true`, Yaguarete reemplaza el catálogo local estático
por el top recomendado por llmfit (en vez de solo reordenar modelos ya existentes).
Además persiste el último catálogo válido para fallback aunque llmfit esté caído.

## Evaluación de Calidad con Logs de Phoenix (Idle)

El ajuste de calidad del router puede analizar `input/output` desde spans en Phoenix:

1. Cada request guarda atributos `yaguarete.input`, `yaguarete.output`, `yaguarete.model_id` y `yaguarete.request_id` en un span.
2. Un scheduler en background espera inactividad (`QUALITY_IDLE_SECONDS`, por defecto `1.0s`).
3. Al cumplirse la ventana idle, consume spans desde Phoenix REST API y evalúa calidad usando el mejor modelo local disponible.
4. Actualiza `effectiveness_score` en `router_stats.db`, impactando futuras decisiones del router.

Variables:

```bash
PHOENIX_API_URL=http://phoenix:6006
OTEL_ENABLED=1
OTEL_BSP_SCHEDULE_DELAY_MS=500
QUALITY_IDLE_SECONDS=1.0
QUALITY_LOOP_INTERVAL_SEC=0.5
PHOENIX_LOOKBACK_SECONDS=600
ROUTER_FEEDBACK_ALPHA=0.35
ENABLE_ADAPTIVE_ALPHA=1
ROUTER_FEEDBACK_ALPHA_MIN=0.05
ROUTER_FEEDBACK_ALPHA_MAX=0.50
ROUTER_FEEDBACK_ALPHA_BASE=0.35
ENABLE_DRIFT_DETECTION=1
ROUTER_DRIFT_THRESHOLD=-0.20
ROUTER_DRIFT_COOLDOWN_MINUTES=15
ROUTER_DRIFT_RECENT_HOURS=2
ROUTER_DRIFT_BASELINE_HOURS=24
ROUTER_DRIFT_MIN_RECENT_SAMPLES=6
ROUTER_DRIFT_MIN_BASELINE_SAMPLES=20
ROUTER_DRIFT_CHECK_INTERVAL_SEC=30
ENABLE_HALLUCINATION_DETECTION=1
ENABLE_HALLUCINATION_LLM_FACT_CHECK=0
TRUST_REMOTE_CODE=0
TRUST_REMOTE_CODE_MODELS=
MODEL_FAILURE_COOLDOWN_SEC=300
MODEL_LOAD_MAX_ATTEMPTS=6
```

También puedes forzar un orden manual de preferencia (solo modelos locales ya definidos en el catálogo):

```bash
YAGUARETE_PREFERRED_LOCAL_MODELS=Qwen/Qwen2.5-Coder-3B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-Coder-1.5B-Instruct
```

Ejemplo para arrancar llmfit como servicio:

```bash
llmfit serve --host 0.0.0.0 --port 8787
```

## Endpoints

- `POST /v1/chat/completions`: Generación de texto (compatible con OpenAI).
- `GET /v1/models`: Listado de modelos disponibles en el catálogo.
- `POST /v1/feedback`: Feedback explícito de respuesta (por `request_id` o `completion_id`).
- `POST /v1/chat/completions/{completion_id}/feedback`: Feedback explícito asociado a una completion.
- `GET /metrics`: Métricas para Prometheus.

### Contrato de Feedback

La respuesta de `POST /v1/chat/completions` devuelve `id=chatcmpl-{request_id}` para correlación directa.

`POST /v1/feedback`

Body (JSON):

```json
{
  "request_id": "9f0cbb04-3552-4dcf-97f8-9aa7f0588fd5",
  "completion_id": "chatcmpl-9f0cbb04-3552-4dcf-97f8-9aa7f0588fd5",
  "score": -1.0,
  "rating": -1,
  "label": "thumbs_down",
  "comment": "No respondió la pregunta",
  "source": "user",
  "user": "jose"
}
```

Reglas:
- Debes enviar `request_id` o `completion_id` (uno de los dos).
- Señal de feedback: `rating` (`-1|0|1`) o `score` (`-1.0..1.0`) o `label`.
- `label` acepta alias (`up/like/positive`, `down/dislike/negative`, `neutral/mixed`).
- Precedencia de señal: `rating` > `score` > `label`.

Response (JSON):

```json
{
  "status": "accepted",
  "request_id": "9f0cbb04-3552-4dcf-97f8-9aa7f0588fd5",
  "completion_id": "chatcmpl-9f0cbb04-3552-4dcf-97f8-9aa7f0588fd5",
  "feedback_score": -1.0,
  "feedback_label": "thumbs_down",
  "effectiveness_old": 0.82,
  "effectiveness_new": 0.53,
  "feedback_alpha": 0.35
}
```

`POST /v1/chat/completions/{completion_id}/feedback`

Body (JSON, igual pero sin `request_id`/`completion_id`):

```json
{
  "rating": 1,
  "comment": "Me sirvió",
  "source": "user",
  "user": "jose"
}
```

## Monitoreo y Observabilidad

Yaguarete expone métricas en `/metrics` para Prometheus. 
Hemos incluido un tablero de Grafana pre-configurado en [grafana_dashboard.json](file:///home/jose/projects/metahumans/yaguarete/grafana_dashboard.json).

Si ejecutas `docker compose`, Phoenix expone por defecto:
- UI host: `16006` -> contenedor `6006`
- OTLP gRPC host: `14317` -> contenedor `4317`
- OTLP HTTP host: `14318` -> contenedor `4318`

Puedes sobrescribirlos con `PHOENIX_UI_PORT`, `PHOENIX_OTLP_GRPC_PORT` y `PHOENIX_OTLP_HTTP_PORT` en tu `.env` para evitar colisiones de puertos.

Si quieres habilitar selección de modelos remotos en el router, usa `ROUTER_ALLOW_REMOTE_MODELS=true`.

### Selección Dinámica de Modelos OpenRouter

Yaguarete puede descubrir modelos remotos en tiempo real desde OpenRouter y priorizarlos con un score combinado de **precio + popularidad**.

Variables opcionales:

```bash
OPENROUTER_DYNAMIC_MODELS=true
OPENROUTER_MODELS_ENDPOINT=https://openrouter.ai/api/v1/models
OPENROUTER_DYNAMIC_MODELS_LIMIT=30
OPENROUTER_DYNAMIC_MODELS_CACHE_TTL_SEC=900
OPENROUTER_DYNAMIC_WEIGHT_PRICE=0.65
OPENROUTER_DYNAMIC_WEIGHT_POPULARITY=0.35
FOUR_BIT_MEMORY_RATIO=0.38
ENABLE_LRU_EVICTION=1
VRAM_ADMISSION_BUFFER_GB=0.35
AUTO_DEVICE_SPLIT_FACTOR=0.5
FORCE_UNLOAD_ALL_FOR_LARGE=0
DISABLE_MULTI_GPU_SPLIT=0
```

Notas:
- Si OpenRouter no responde, Yaguarete usa el catálogo remoto estático como fallback.
- `OPENROUTER_DYNAMIC_WEIGHT_PRICE` y `OPENROUTER_DYNAMIC_WEIGHT_POPULARITY` se normalizan automáticamente.
- La caché evita consultar `/models` en cada request y reduce latencia.
- `FOUR_BIT_MEMORY_RATIO` ajusta la estimación efectiva de vRAM cuando se usa cuantización 4-bit.
- `ENABLE_LRU_EVICTION` habilita descarga de modelos menos usados cuando falta vRAM.
- `VRAM_ADMISSION_BUFFER_GB` reserva margen para evitar OOM por picos.
- `DISABLE_MULTI_GPU_SPLIT=1` desactiva la distribución automática de modelos grandes entre GPUs (si experimentas errores de "tensors on different devices", actívalo).

Para usarlo:
1. En Grafana, ve a **Dashboards** > **Import**.
2. Sube el archivo `grafana_dashboard.json` o pega su contenido.
3. Selecciona tu datasource de **Prometheus**.

---
Para más detalles sobre el algoritmo de enrutamiento, consulta [router_documentation.md](file:///home/jose/.gemini/antigravity/brain/c6e11bcb-8590-4413-8313-385a5a46575e/router_documentation.md).
