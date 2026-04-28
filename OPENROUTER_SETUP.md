# OpenRouter Integration for Yaguarete

## Overview

Yaguarete ahora está integrado con OpenRouter para acceder a modelos remotos de alta calidad. Esto permite que el router pueda seleccionar entre:

- **Modelos Locales**: Qwen 2.5 (1.5B, 7B, 32B) para máxima privacidad y bajo costo de infraestructura
- **Modelos Remotos (OpenRouter)**: Gpt-4o-mini, Claude 3.5 Sonnet, Llama 3.1, etc. para máxima calidad cuando sea necesario

## Modelos Disponibles

### Locales (sin costo de API)
```
Qwen/Qwen2.5-1.5B-Instruct     | SMALL  | CHAT     | $0.00005/1k chars
Qwen/Qwen2.5-7B-Instruct       | MEDIUM | CHAT     | $0.0002/1k chars
Qwen/Qwen2.5-Coder-1.5B        | SMALL  | CODE     | $0.00005/1k chars
Qwen/Qwen2.5-Coder-7B          | MEDIUM | CODE     | $0.0002/1k chars
Qwen/Qwen2.5-32B-Instruct      | LARGE  | REASONING| $0.001/1k chars
```

### Remotos via OpenRouter (mayor calidad)
```
meta-llama/llama-3.1-8b-instruct:free | SMALL  | CHAT     | FREE (~$0.0/1k chars)
openai/gpt-4o-mini                    | MEDIUM | CHAT     | $0.000075/1k chars
anthropic/claude-3.5-sonnet           | LARGE  | REASONING| $0.00375/1k chars
openai/gpt-4-turbo                    | LARGE  | REASONING| $0.002/1k chars
```

## Configuración

### 1. Obtener API Key de OpenRouter

1. Ve a https://openrouter.ai/keys
2. Inicia sesión o crea una cuenta
3. Copia tu API key

### 2. Configurar Variables de Entorno

La API key ya se utiliza del backend de historiathor. Para Docker:

```bash
export OPENROUTER_API_KEY="sk-or-..."
docker-compose up -d
```

O crea un archivo `.env` en la carpeta yaguarete:

```bash
cp .env.example .env
# Edita .env con tu OPENROUTER_API_KEY
docker-compose --env-file .env up -d
```

### 3. Verificar Configuración

```bash
# Listar modelos disponibles
curl http://localhost:8001/v1/models | jq

# Verificar salud
curl http://localhost:8001/health
```

## Cómo Funciona el Routing

Yaguarete usa un **router inteligente (k-NN semántico)** que automáticamente:

1. **Analiza la solicitud** con embeddings
2. **Busca en historial** de requests anteriores similares
3. **Calcula score** para cada modelo basado en:
   - Similitud semántica (qué model funcionó antes para requests parecidas)
   - Costo (preferencia por modelos baratos)
   - Duración (preferencia por respuestas rápidas)
   - Calidad histórica (requiere evaluación posterior)

4. **Selecciona el mejor modelo** balanceando:
   - **Efectividad máxima**: Usa modelos más potentes (Claude, GPT-4) cuando es necesario
   - **Costo mínimo**: Prefiere modelos locales o Llama 3.1 free cuando es suficiente

### Estrategia de Costo Óptimo

El router mantiene estadísticas de:
- Qué modelos generan mejor calidad para cada tipo de tarea
- El costo de cada generación
- El tiempo de respuesta

Con esto, tiende a:
- Usar **Qwen 1.5B (local)** para tareas simples
- Usar **Llama 3.1 free** para tareas moderadas
- Usar **gpt-4o-mini** solo cuando la tarea requiere calidad media-alta
- Usar **Claude 3.5 Sonnet** solo para tareas que requieren razonamiento complejo

## Endpoint API

Compatible con OpenAI API format:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yaguarete/auto",  # O especific un model ID
    "messages": [
      {"role": "user", "content": "Hola, ¿cómo estás?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Integración con Backend

Para cambiar el backend a usar Yaguarete en lugar de OpenRouter directo, solo cambiar en `backend/src/ai/ai.service.ts`:

```typescript
// De:
private readonly apiUrl = 'https://openrouter.ai/api/v1/chat/completions';

// A:
private readonly apiUrl = 'http://yaguarete:8001/v1/chat/completions';
```

Los modelos y la API key ya estarían centralizados en Yaguarete.

## Monitoring y Observabilidad

Yaguarete proporciona:
- **Prometheus metrics** en `/metrics`:
  - Requests por modelo
  - Tiempo de respuesta
  - Costo acumulado
  - Calidad de respuestas

- **Distributed tracing** via Phoenix (puerto 6006)
  - Ver traces de cada request
  - Identificar cuellos de botella

- **SQLite stats** en `data/router_stats.db`:
  - Historial completo de requests
  - Análisis de efectividad por modelo
  - Tendencias a lo largo del tiempo

## Troubleshooting

### "OPENROUTER_API_KEY is not defined"
```bash
echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" > yaguarete/.env
docker-compose --env-file .env up -d yaguarete
```

### Modelos locales no cargan
Verifica que tengas suficiente VRAM y espacio en disco. El router fallará gracefully al modelo SMALL local.

### OpenRouter API returns 401
Verifica que tu API key sea válida en https://openrouter.ai/keys

## Costos Esperados

Con la estrategia actual:
- **Uso ligero**: ~$0.01/día con local models
- **Uso moderado**: ~$0.10-$0.50/día (mayoría Llama free + algunos gpt-4o-mini)
- **Uso pesado**: ~$1-$5/día (mezcla con Claude 3.5)

Los modelos locales consumen energía de GPU pero NO cuestan API.
