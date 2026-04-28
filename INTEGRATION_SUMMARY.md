# Integración OpenRouter en Yaguarete - Resumen de Cambios

## Cambios Realizados

### 1. **Nuevo Cliente OpenRouter** 
Archivo: `src/infrastructure/transformers_engine/openrouter_client.py`
- Cliente para hacer llamadas a OpenRouter API v1/chat/completions
- Manejo de errores y timeouts
- Verificación de disponibilidad de API

### 2. **Catálogo de Modelos Mejorado**
Archivo: `src/infrastructure/transformers_engine/model_catalog.py`

**Enumeraciones nuevas:**
- `ModelProvider`: LOCAL / OPENROUTER
- Campo `is_remote` en ModelDefinition para facilitar filtrado

**Modelos añadidos (10 → 14 modelos):**

**OPENROUTER (4 nuevos):**
```python
meta-llama/llama-3.1-8b-instruct:free    # FREE tier
openai/gpt-4o-mini                       # Ideal MEDIUM: $0.000075/1k chars
anthropic/claude-3.5-sonnet              # Mejor LARGE (reasoning)
openai/gpt-4-turbo                       # Premium LARGE
```

**Métodos nuevos para búsqueda:**
- `find_best_local_model()`: Solo modelos locales
- `find_best_remote_model()`: Solo OpenRouter (optimiza por costo)
- `get_all_local_models()`: Lista todos locales
- `get_all_remote_models()`: Lista todos remotos

### 3. **Router Inteligente para Costo Óptimo**
Archivo: `src/infrastructure/transformers_engine/model_router.py`

**Nueva estrategia `_select_cost_optimal_model()`:**
```
1. Intenta modelo local primero (si PREFER_LOCAL_MODELS=true)
2. Si no disponible, usa FREE tier (Llama 3.1)
3. Para MEDIUM: gpt-4o-mini ($0.000075/1k - muy barato)
4. Para LARGE: Claude 3.5 Sonnet (mejor reasoning)
```

**Variable de entorno:**
- `PREFER_LOCAL_MODELS=true` (por defecto)

### 4. **Models Handler Actualizado**
Archivo: `src/infrastructure/transformers_engine/models_handler.py`

**Cambios:**
- Import de `OpenRouterClient`
- Lazy initialization: `self._openrouter_client`
- Método `_get_openrouter_client()` para obtener cliente
- Lógica en `generate_text()` para detectar modelos remotos:
  - Si `model_def.is_remote == True`: usa OpenRouter
  - Si falla OpenRouter: fallback a modelos locales
  - Manejo de errores con graceful degradation

### 5. **Configuración Docker**
Archivo: `docker-compose.yml`

**Variables de entorno nuevas:**
```yaml
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
OPENROUTER_HTTP_REFERER=http://localhost:8001
OPENROUTER_X_TITLE=Yaguarete LLM Proxy
```

### 6. **API Endpoint Mejorado**
Archivo: `src/main.py` - Endpoint `/v1/models`

**Ahora incluye:**
```json
{
  "id": "openai/gpt-4o-mini",
  "is_remote": true,
  "provider": "openrouter",
  "cost_per_1k_chars": 0.000075,
  "estimated_vram_gb": 0.0,
  "complexity": "medium",
  "specialty": "chat"
}
```

### 7. **Archivos de Configuración**
- `.env`: Contiene OPENROUTER_API_KEY (copiada del backend)
- `.env.example`: Template para documentación
- `OPENROUTER_SETUP.md`: Setup y troubleshooting

## Estrategia de Costo Óptimo (Máxima Efectividad, Mínimo Costo)

### Matriz de Selección

| Complejidad | Preferencia | Costo Estimado |
|---|---|---|
| **SMALL** | Qwen 1.5B local | ~$0.00005/1k |
| **MEDIUM** | Qwen 7B local → Llama 3.1 free → gpt-4o-mini | $0.00005-$0.000075/1k |
| **LARGE** | Qwen 32B local → Claude 3.5 Sonnet | $0.001-$0.00375/1k |

### Lógica de Decisión

**Para SMALL (tareas simples, chatbots, FAQ):**
```
1. Qwen 1.5B local (gratis, rápido, MUY bueno para tareas simples)
2. Si no disponible: Llama 3.1 free
```

**Para MEDIUM (escritura creativa, ayuda general):**
```
1. Qwen 7B local (15GB VRAM, excelente balance)
2. Si GPU ocupada: Llama 3.1 free (<$0.000001/1k)
3. Si necesita mejor calidad: gpt-4o-mini ($0.000075/1k, muy barato)
```

**Para LARGE (razonamiento, análisis complejo):**
```
1. Qwen 32B local (si tenemos 68GB VRAM)
2. Si OOM: Claude 3.5 Sonnet ($0.00375/1k, mejor reasoning)
3. Fallback: gpt-4o-mini como último recurso
```

### Resultados Esperados

**Escenario típico (uso moderado):**
- 70% → Modelos locales (Qwen) = $0
- 20% → Llama 3.1 free = ~$0.00002
- 10% → gpt-4o-mini = ~$0.00005
- **Total: ~$0.00007 por 1000 chars = $70 por 1 MILLÓN de chars**

**Con 1000 requests/día de 500 chars promedio:**
- 500 chars × 1000 = 500,000 chars/día
- Costo estimado: $0.035/día = ~$1/mes

## Testing y Verificación

### 1. Verificar sintaxis
```bash
cd yaguarete
python3 -m py_compile src/infrastructure/transformers_engine/{openrouter_client,model_catalog,model_router}.py
```

### 2. Listar modelos disponibles
```bash
# Asegúrate de que yaguarete está corriendo
curl http://localhost:8001/v1/models | jq '.' | grep -E '"id"|"is_remote"|"cost"'
```

### 3. Test de request a modelo local
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hola"}]
  }'
```

### 4. Test de request a OpenRouter
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Explica quantum computing"}]
  }'
```

### 5. Test de auto-routing (el router elige el mejor modelo)
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yaguarete/auto",
    "messages": [{"role": "user", "content": "Escribe una función Python para calcular fibonacci"}]
  }'
```

## Próximos Pasos (cuando redirijas el backend)

Para usar Yaguarete en lugar de OpenRouter directo en el backend:

### `backend/src/ai/ai.service.ts`
```typescript
// Cambiar:
private readonly apiUrl = 'https://openrouter.ai/api/v1/chat/completions';

// Por:
private readonly apiUrl = 'http://yaguarete:8001/v1/chat/completions';

// Y opcional: cambiar modelo
model: 'yaguarete/auto'  // O un modelo específico
```

Esto centralizaría:
- **API key**: Manejada por Yaguarete
- **Routing inteligente**: Elegir entre local/OpenRouter automáticamente
- **Estadísticas**: Consolidadas en Yaguarete
- **Costos**: Optimizados por el router

## Troubleshooting

### "OPENROUTER_API_KEY is not set"
```bash
# Verificar que .env existe
ls -la yaguarete/.env

# Recargar containers
docker-compose down
docker-compose --env-file yaguarete/.env up -d yaguarete
```

### Modelos remotos retornan error 401
```bash
# Verificar API key válida en https://openrouter.ai/account/billing/limits
cat yaguarete/.env | grep OPENROUTER_API_KEY
```

### Modelos locales no se cargan
```bash
# Ver logs
docker logs yaguarete-llm-proxy | grep -i "qwen\|load\|error"

# Verificar GPU disponibilidad
docker logs yaguarete-llm-proxy | grep "GPU\|CUDA"
```
