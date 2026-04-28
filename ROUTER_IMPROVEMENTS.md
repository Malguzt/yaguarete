# Mejoras Estratégicas: Router y Análisis Posmortem de Retroalimentación

## Análisis Actual

### Fortalezas
✅ **Router k-NN semántico**: Usa embeddings para encontrar contextos históricos similares
✅ **Evaluación multidimensional**: Format, density, judge (relevancia), sentiment
✅ **Retroalimentación explícita**: Sistema de calificación manual (rating, score, label)
✅ **Feedback scheduler**: Evaluación automática en ventanas inactivas
✅ **Persistencia SQLite**: Trazabilidad completa con WAL mode

### Debilidades Identificadas

1. **Router**:
   - ❌ Penalización de sesión es binaria (solo si similitud > 0.85)
   - ❌ No hay clustering de tópicos para evitar repetir errores
   - ❌ Sin historial de "modelos fallidos" por tipo de pregunta
   - ❌ Score final mezcla effectiveness + cost pero sin normalización dinámica
   - ❌ Sin adaptación temporal (drift de calidad no se detecta)
   - ❌ Metadata de OpenRouter no se integra con histórico local

2. **Evaluación de Calidad**:
   - ❌ Judge_score depende de modelo pequeño (puede ser impreciso)
   - ❌ No hay detección de alucinaciones (hallucinations)
   - ❌ Sentiment de 3 clases es muy simplista
   - ❌ Format check solo básico (JSON, código)
   - ❌ Sin evaluación de consistencia con contexto anterior

3. **Feedback Posmortem**:
   - ❌ Solo evalúa respuesta en aislamiento (sin context)
   - ❌ Sin correlation entre feedback explícito y automático
   - ❌ Alpha exponencial (0.35) es fijo, no se adapta
   - ❌ Sin detección de feedback contradictorio
   - ❌ Sin análisis de causas raíz de fallos

4. **Observabilidad**:
   - ❌ Sin alertas sobre degradación de calidad
   - ❌ Sin análisis de tendencias (time series)
   - ❌ Sin breakdown por complejidad/especialidad
   - ❌ Sin A/B testing framework

---

## Mejoras Propuestas (Prioridad)

### 🔴 CRÍTICAS (Semana 1-2)

#### 1. **Hallucination Detection** (QualityEvaluator)
```python
def _detect_hallucinations(self, prompt: str, response: str, analysis_model_id: str) -> float:
    """
    Detects if response contains made-up facts/claims.
    Uses local LLM as fact-checker or regex patterns for specific domains.
    Returns score 0.0-1.0 (1.0 = confident no hallucination)
    """
```

**Beneficio**: Evita que modelos "inventadores" se sigan seleccionando
**Implementación**: 
- Pattern matching para "Según" / "Dice que" sin contexto
- Segunda llamada LLM pidiendo fuentes
- Comparación de hechos con prompt original

#### 2. **Dynamic Alpha Adjustment** (RouterStatsRepository)
```python
def get_adaptive_feedback_alpha(self, model_id: str) -> float:
    """
    Alpha varía según:
    - Confianza en histórico (más data = menos alpha)
    - Varianza de scores (alta varianza = menos alpha)
    - Recency (eventos recientes pesan más)
    """
    # Alpha = base * (1 - min(sample_count, 100) / 100) * (1 + variance_factor)
```

**Beneficio**: Feedback converge más rápido, se adapta a cambios
**Implementación**: Ya tienes las métricas, solo falta el cálculo dinámico

#### 3. **Temporal Decay & Drift Detection** (RouterService)
```python
def detect_quality_drift(self, model_id: str, window_hours: int = 24) -> float:
    """
    Compara effectiveness reciente vs histórico.
    Retorna delta (positivo = mejora, negativo = degradación)
    Si delta < -0.2, descarta modelo automáticamente por 1 hora
    """
```

**Beneficio**: Detecta automáticamente cuando modelos se vuelven inestables
**Implementación**: Agregar columna `quality_outlier_flag` a model_stats

---

### 🟠 IMPORTANTES (Semana 3-4)

#### 4. **Multi-Dimensional Feedback Correlation** (RouterStatsRepository)
```python
def correlate_feedback_signals(self, request_id: str) -> dict:
    """
    Correlaciona:
    - Feedback explícito (user rating) vs automático (judge_score)
    - Feedback vs tiempo (¿cambió el rating después de cierta latencia?)
    - Patrón: "Le doy 5 estrellas pero 30s después dije que no sirvió"
    """
```

**Beneficio**: Detecta feedback contradictorio, identifica patrones de usuario
**Implementación**: 
- Nuevo índice en model_stats: `feedback_timestamp`
- Correlación Pearson entre señales

#### 5. **Topic Clustering & Failure Patterns** (RouterService)
```python
def get_topic_cluster(self, embedding: List[float]) -> str:
    """
    Agrupa requests similares en "tópicos" (LDA o similitud centroide).
    Mantiene matriz de "modelos fallidos por tópico":
    {
        "math": {model_A: 0.15_fail_rate, model_B: 0.85_fail_rate},
        "code": {model_A: 0.40_fail_rate, model_B: 0.10_fail_rate}
    }
    """
```

**Beneficio**: Router especializado por dominio, evita modelos malos en contexto X
**Implementación**: 
- Clustering batch cada N requests
- Cache de 10-15 tópicos principales

#### 6. **Advanced Sentiment Analysis** (QualityEvaluator)
```python
def _analyze_sentiment_detailed(self, text: str, analysis_model_id: str) -> dict:
    """
    Retorna:
    - sentiment: -1, 0, 1
    - confidence: 0.0-1.0
    - emotion_tags: ["frustración", "confusión", "satisfacción"]
    - language_markers: {"exclamaciones": 2, "interrogaciones": 5}
    """
```

**Beneficio**: Comprensión más rica del feedback, detecta frustración implícita
**Implementación**: Expandir _check_sentiment actual

---

### 🟡 MEJORAS OPERACIONALES (Semana 5+)

#### 7. **A/B Testing Framework** (main.py + new file)
```python
class ABTestingFramework:
    """
    Permite comparar 2 routers en paralelo:
    - Router A (v1): actual
    - Router B (v2): con nuevos heurísticos
    Métrica: effectiveness_score, cost, duration
    """
```

**Beneficio**: Valida mejoras antes de deployarlas
**Implementación**: Split requests 50/50, track separadamente

#### 8. **Root Cause Analysis Dashboard** (new file)
```python
class PostmortemAnalyzer:
    """
    Análisis automático de fallos:
    1. Clustera requests fallidas
    2. Encuentra patrón común (complejidad, tópico, modelo)
    3. Sugiere acción (reentrenar, cambiar modelo, actualizar prompt)
    4. Expone en Prometheus como métrica
    """
```

**Beneficio**: De "el modelo X falló" a "X falla en preguntas sobre fechas cuando..."
**Implementación**: GraphQL query sobre stats, pattern mining

#### 9. **Context Window Optimization** (RouterService)
```python
def optimize_model_for_context(self, session_history: List[str]) -> str:
    """
    Analiza conversación previa:
    - Si hay mucho contexto: prefiere LARGE models (tienen max_tokens > 2048)
    - Si es primera pregunta: prefiere SMALL (latencia < 1s)
    - Si hay muchas repreguntasss (>3): penaliza modelo anterior
    """
```

**Beneficio**: Router contextual, no solo por prompt actual
**Implementación**: Usar `session_history` table que ya existe

#### 10. **Feedback Loop Closure** (main.py)
```python
@app.post("/v1/feedback/{completion_id}/explain")
async def explain_feedback_impact(completion_id: str) -> dict:
    """
    Cuando user da feedback, retorna:
    - Cómo afectó el score de efectividad
    - Qué modelos se vieron impactados
    - Cuándo se verán cambios (próximas N requests)
    - Sugerencias para mejorar (ej: "Sé más específico")
    """
```

**Beneficio**: Cierra el loop, educauser sobre qué feedback es útil
**Implementación**: Query sobre router_stats, mostrar impacto real

---

## Roadmap de Implementación

### Fase 1 (Semana 1-2): Foundation
- [ ] Hallucination detection
- [ ] Dynamic alpha
- [ ] Temporal drift detection
- [ ] Tests & metrics

### Fase 2 (Semana 3-4): Intelligence
- [ ] Feedback correlation
- [ ] Topic clustering
- [ ] Advanced sentiment
- [ ] Root cause analysis

### Fase 3 (Semana 5+): Optimization
- [ ] A/B testing
- [ ] Context optimization
- [ ] Feedback loop closure
- [ ] Dashboard integration

---

## Métricas de Éxito

| Métrica | Baseline | Target (8 semanas) |
|---------|----------|-------------------|
| Effectiveness avg | 0.65 | 0.82+ |
| Model selection accuracy | 60% | 85%+ |
| Hallucination rate | ?% | < 5% |
| Feedback utilization | 40% | 90%+ |
| Time to recovery (drift) | ∞ | < 15 min |
| Cost per request | $0.05 | $0.03 |

---

## Quick Wins (Sin arquitectura nueva)

1. **Pescar feedback contradictorio**: Flag si rating=5 pero sentiment=NEGATIVE
2. **Exponential smoothing**: Replace simple average con EMA en effectiveness
3. **Session penalty decay**: Si repregunta, penaliza modelo anterior solo esta sesión
4. **Cost normalization**: Divide effectiveness por (1 + cost_per_1k_chars)
5. **Embed quality scores**: Incluir format/judge en embedding decision, no solo en retrospect

---

## Stack Sugerido para Implementación

- **Query**: SQLAlchemy ORM (mejor que raw SQL) → migration a PostgreSQL después
- **ML**: scikit-learn (KMeans/LDA para clustering), scipy (correlación)
- **Monitoring**: Expandir Prometheus (nuevas métricas), Grafana (nuevo dashboard)
- **Testing**: pytest + hypothesis para generative testing
- **Dashboard**: Streamlit para exploración rápida de stats

