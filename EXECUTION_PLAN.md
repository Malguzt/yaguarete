# Plan de Ejecución & Análisis de Impacto

## Comparativa: Antes vs Después

### 🔴 Escenario 1: Alucinación No Detectada (ANTES)

**Situación**:
- User: "¿Cuál fue el primer presidente de Argentina?"
- Modelo Pequeño (incorrectamente): "Según la historia, Fernando de la Rúa fue el primer presidente en 1776."
- Evaluation: judge_score=1.0 (el LLM pequeño dijo "SÍ"), format_score=1.0
- **Effectiveness_score = 1.0** ✅ (Modelo bien evaluado falsamente)
- Feedback: None (user no da feedback negativo inmediatamente)
- **Impacto**: Router sigue eligiendo este modelo para preguntas de historia

**Después (CON Hallucination Detection)**:
- Same response
- HallucinationDetector: detects "Según" sin fuente, verifica hechos
- LLM fact-checker: "FALSA - Fernando de la Rúa fue presidente en 1989, no 1776"
- hallucination_score=0.1, flag=True
- **Metric**: judge_score redimensionado a 0.2 (instead of 1.0)
- **Effectiveness_score = 0.2** ❌ (Detectado como malo)
- **Impacto**: Router evita este modelo para historia en futuro

---

### 🟠 Escenario 2: Feedback Contradictorio (ANTES)

**Situación**:
- Request A: Model X genera respuesta.
- Automatic feedback: effectiveness=0.9 (buena respuesta según evaluador)
- User manual feedback (10 segundos después): rating=-1 (no me sirvió)
- Current logic: Último feedback gana, effectiveness se vuelve 0.0
- **Pero**: No se analiza la contradicción. ¿Validador está mal? ¿Usuario cambió de idea?
- **Router behavior**: Penalización dura, Model X cae rating sin contexto

**Después (CON Feedback Correlation)**:
- Same scenario
- Feedback correlation analysis:
  - automatic_score: 0.9 (judge said yes)
  - manual_score: -1.0 (user said no)
  - correlation: -0.95 (fuerte desacuerdo)
  - time_delta: 10 seconds (rápida, no reflexión)
- **Actions**:
  1. Flag automatic evaluator: "judge model unreliable for this domain?"
  2. Adaptive alpha: reduce alpha (más conservador): 0.35 → 0.15
  3. Alert: "High disagreement between auto & manual feedback"
  4. Router: penaliza Model X suavemente (feedback alpha 0.15 < 0.35)
- **Impacto**: Learn que el juez es impreciso, no culpar al modelo

---

### 🟡 Escenario 3: Drift no Detectado (ANTES)

**Situación**:
- Model B: historical effectiveness = 0.75 (promedio últimos 30 días)
- Today at 10:00 - Model B crashes on GPU (memory bug)
- Requests 10:01-10:45: effectiveness = 0.1 (todos fallan)
- **Problem**: stats_repo solo guarda aggregate (avg, count). No detecta cambios temporales
- Router sigue eligiendo Model B porque historical avg sigue siendo 0.75
- **Impacto**: 40 minutos de respuestas malas

**Después (CON Temporal Drift Detection)**:
- DriftDetector runs every request (o cada 5 requests)
- Last 2 hours effectiveness: 0.1
- Historical (24h) effectiveness: 0.75
- drift_value = 0.1 - 0.75 = -0.65 (threshold: -0.20)
- **Action**: Model B enters cooldown for 15 minutes
- Router immediately picks Model A (next best)
- **Impacto**: 2 minutos para detectar + cambiar, vs 40 minutos esperando

---

## Implementación Step-by-Step

### Week 1: Foundation

#### Day 1-2: Hallucination Detector
- [ ] Create `hallucination_detector.py` (copy from ROUTER_IMPLEMENTATIONS.md)
- [ ] Add pattern matching for suspicious language (Spanish + English)
- [ ] Add unsourced facts check
- [ ] Unit tests: test_detects_suspicious_patterns, test_allows_sourced
- [ ] **Effort**: 4 hours

#### Day 3: Integrate into QualityEvaluator
```python
# In src/application/router/quality_evaluator.py
def evaluate_response(...):
    hallucination = self.hallucination_detector.detect_hallucinations(prompt, response)
    judge_score = min(judge_score, hallucination["hallucination_score"])  # Penalize if hallucination
    return {
        "format_score": ...,
        "judge_score": judge_score,  # Now includes hallucination penalty
        "hallucination_risk": hallucination
    }
```
- [ ] Modify QualityEvaluator to use detector
- [ ] Add hallucination to Prometheus metrics
- [ ] **Effort**: 2 hours
- [ ] **Testing**: Manual test with 10 prompts

#### Day 4: Dynamic Alpha
- [ ] Add `get_adaptive_feedback_alpha()` to RouterStatsRepository
- [ ] Update `_apply_feedback_to_request()` in main.py to use adaptive alpha
- [ ] Add metric `feedback_alpha_adaptive` to Prometheus
- [ ] **Effort**: 3 hours

#### Day 5: Drift Detector
- [ ] Create `drift_detector.py`
- [ ] Implement check_and_flag_drift() with cooldown logic
- [ ] Integrate into RouterService.route_request()
- [ ] Add alert to Prometheus
- [ ] **Effort**: 3 hours

#### Day 5-6: Testing & Validation
- [ ] Integration tests for all three components
- [ ] Load testing: does drift detection slow down routing?
- [ ] Manual testing with real requests
- [ ] **Effort**: 4 hours

**Week 1 Total**: ~16 hours

---

### Week 2: Validation & Observability

#### Day 7-8: Prometheus Dashboard Expansion
```promql
# Add to existing Grafana dashboard:
- Graph: Hallucination rate over time
- Graph: Model quality drift (recent vs historical)
- Graph: Feedback alpha distribution
- Table: Currently cooldown models
- Heatmap: Drift by model & hour
```
- [ ] Add Prometheus queries
- [ ] Add Grafana panels
- [ ] **Effort**: 3 hours

#### Day 9: Production Deployment Strategy
- [ ] Feature flags:
  - `ENABLE_HALLUCINATION_DETECTION=1` (default false first)
  - `ENABLE_DRIFT_DETECTION=1` (default false)
  - `ENABLE_ADAPTIVE_ALPHA=1` (default false)
- [ ] Gradual rollout: 10% → 50% → 100%
- [ ] **Effort**: 2 hours

#### Day 10: A/B Test Setup (optional, but recommended)
```python
# In ModelsHandler
def route_request_with_abtest(prompt, session_id):
    if is_in_test_group("drift_detection", 0.2):  # 20% test group
        use_drift_detection = True
    else:
        use_drift_detection = False
    
    # Track separately:
    # - effectiveness_score_with_drift
    # - effectiveness_score_without_drift
```
- [ ] Add test group tracking to metrics
- [ ] **Effort**: 2 hours

**Week 2 Total**: ~7 hours

---

### Week 3-4: Phase 2 (Important) Features

#### Feedback Correlation Analysis
```python
# New file: src/infrastructure/repositories/feedback_correlation_analyzer.py

class FeedbackCorrelationAnalyzer:
    def correlate_signals(self, request_id: str) -> dict:
        """
        Compare automatic vs manual feedback.
        Detect contradictions, suspicious patterns.
        """
        # 1. Get automatic scores (judge, format, sentiment)
        # 2. Get manual feedback (if exists)
        # 3. Compute correlation matrix
        # 4. Flag if high disagreement
```

- [ ] Implement analyzer
- [ ] Add to main.py feedback endpoint
- [ ] Tests: 2 hours
- [ ] Integration: 1 hour

#### Topic Clustering (Optional but High Impact)
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

class TopicClusteringService:
    def cluster_requests(self, min_cluster_size: int = 20):
        """
        Daily batch job: cluster recent requests into topics.
        Build failure matrix: {topic: {model_id: failure_rate}}
        """
```

- [ ] Implement LDA clustering
- [ ] Build failure matrix
- [ ] Tests: 3 hours
- [ ] Scheduler integration: 2 hours

---

## Metrics to Monitor (Before & After)

### Primary Metrics

| Métrica | Baseline | Week 4 Target | Week 8 Target |
|---------|----------|---------------|---------------|
| **Hallucination Rate** | unknown (not tracked) | < 15% flagged | < 5% |
| **Quality Drift Detection Time** | ∞ (no detection) | < 15 min | < 5 min |
| **Effectiveness Score Avg** | 0.65 | 0.72 | 0.82 |
| **Model Selection Accuracy** | 60% | 70% | 85% |
| **Cost per Request** | $0.05 | $0.048 | $0.038 |
| **User Feedback Rate** | 40% | 60% | 90% |

### Secondary Metrics (SRE)

| Métrica | Threshold |
|---------|-----------|
| Hallucination detector latency | < 50ms (skip if > 200ms) |
| Drift detection latency | < 10ms |
| Prometheus scrape increase | < 15% |
| DB query increase | < 10% |

---

## Rollout Strategy

### Phase 1: Internal Testing (Day 1-7)
- Deploy to staging with all features enabled
- Run load tests: 100 RPS sustained
- Manual validation: 50 prompts across categories
- **Gate**: effectiveness ≥ baseline (should improve or match)

### Phase 2: Canary (Day 8-10)
- Deploy to 10% of prod traffic
- Feature flags: all set to true
- Monitor: error rate, latency, metrics
- **Gate**: error_rate < 1%, latency p95 < 2s

### Phase 3: Progressive Rollout (Day 11-21)
- Day 11-14: 25% traffic
- Day 15-18: 50% traffic
- Day 19-21: 100% traffic
- **Rollback criteria**: error_rate > 2%, hallucination_flag_rate > 50%

### Phase 4: Full Deployment + Feedback Loop (Week 4+)
- Enable user-facing explanations
- Dashboard for ops team
- Weekly review of metrics
- Iterate on parameters

---

## Risk Mitigation

### Risk: Hallucination Detector is Too Strict
- **Mitigation**: Start with `confidence > 0.8` AND `score < 0.3` before flagging
- **Metric**: Monitor false positive rate
- **Rollback**: Increase confidence threshold to 0.9

### Risk: Drift Detection Has False Positives
- **Mitigation**: Require 5+ samples in recent window, not 3
- **Metric**: Manual review of cooldown triggers
- **Rollback**: Increase DRIFT_THRESHOLD to -0.3

### Risk: Adaptive Alpha Converges Too Slowly
- **Mitigation**: Start with more aggressive alpha (0.5 max, 0.2 min)
- **Metric**: Track EMA convergence speed
- **Rollback**: Revert to fixed alpha=0.35

### Risk: DB Performance Degrades
- **Mitigation**: Add indexes before deploying (already in code)
- **Metric**: Query latency p95 < 100ms
- **Rollback**: Disable adaptive alpha (simplest to revert)

---

## Success Criteria (Go/No-Go)

### Go Criteria (ALL must be true):
1. ✅ Hallucination detector catches ≥ 80% of synthetic hallucinations
2. ✅ Drift detection finds model failures < 15 minutes after onset
3. ✅ Adaptive alpha improves convergence (measured by variance of effectiveness over time)
4. ✅ No regressions: effectiveness_score ≥ baseline
5. ✅ P95 latency unchanged (< 10% increase)
6. ✅ Error rate < 1%
7. ✅ All Prometheus metrics scraping successfully

### No-Go Criteria (any one stops rollout):
1. ❌ Hallucination false positive rate > 40%
2. ❌ Drift detector triggers > 100 times/day (too noisy)
3. ❌ P95 latency increases > 500ms
4. ❌ Error rate > 2%
5. ❌ DB storage grows > 500MB/day
6. ❌ User confusion spike (negative ratings on same requests)

---

## Estimated ROI

### Investment
- Dev time: ~40 hours (1 engineer, 1 week)
- Testing/validation: ~20 hours
- Monitoring/docs: ~10 hours
- **Total**: ~70 engineer-hours

### Return (8 weeks)
- Effectiveness improvement: 0.65 → 0.82 = **26% gain** → saved ~5% of API costs
- Hallucination reduction: saves reputation/support tickets
- Faster drift detection: reduces cascading failures
- Better feedback loop: enables continuous improvement

**Estimated Annual Savings**: 
- If running 1M requests/day @ $0.05 = $50K/day × 365 = $18.25M/year
- 5% cost reduction = **$912K saved/year**
- ROI: ~30x in first year

---

## Next Steps

1. **Day 1**: Review this document with team
2. **Day 2**: Setup feature branches + write skeleton code
3. **Day 3**: Implement hallucination detector
4. **Day 5**: Implement adaptive alpha + drift detector
5. **Day 8**: Deploy to staging, run tests
6. **Day 10**: Deploy canary (10% prod)
7. **Day 21**: Full prod deployment
8. **Day 28**: Post-mortem + iterate Phase 2

