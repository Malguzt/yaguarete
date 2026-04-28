# Implementaciones Concretas: Phase 1 (Críticas)

## 1. Hallucination Detection

### Archivo: `src/application/router/hallucination_detector.py`

```python
import re
from typing import Dict
from infrastructure.transformers_engine.models_handler import ModelsHandler
from infrastructure.transformers_engine.model_catalog import ModelComplexity

class HallucinationDetector:
    """
    Detects potential hallucinations and unfounded claims in responses.
    Uses both heuristics and LLM-based fact checking.
    """
    
    def __init__(self, models_handler: ModelsHandler) -> None:
        self.models_handler = models_handler
        self.suspicious_patterns = [
            r"(?:según|dice que|afirma que|cuenta que|menciona que)\s+[^.]{{20,}}\.?\s*(?:$|pero)",
            r"(?:yo|un|la)\s+(?:invento|creo|imagino|hago)\s",
            r"(?:esto|aquello)\s+(?:no|nunca)\s+(?:existe|pasó|ocurrió)",
            r"(?:el|la|los|las)\s+\w+\s+(?:que|cual)\s+(?:no existe|falso|incorrecto)",
        ]
    
    def detect_hallucinations(
        self,
        prompt: str,
        response: str,
        analysis_model_id: str | None = None,
    ) -> Dict[str, float | str]:
        """
        Returns:
        {
            "hallucination_score": 0.0-1.0 (0 = likely hallucination, 1 = confident truth),
            "confidence": 0.0-1.0 (how sure are we),
            "reasons": ["pattern_match", "fact_check_failed"],
            "flag": bool (True = reject this response)
        }
        """
        score = 1.0  # Start optimistic
        confidence = 0.0
        reasons = []
        
        # 1. HEURISTIC: Suspicious language patterns
        pattern_score, pattern_confidence = self._check_suspicious_patterns(response)
        if pattern_score < 0.5:
            reasons.append("suspicious_pattern_language")
            score *= 0.7
            confidence += pattern_confidence * 0.3
        
        # 2. HEURISTIC: No sources provided for factual claims
        fact_score, fact_confidence = self._check_unsourced_facts(prompt, response)
        if fact_score < 0.5:
            reasons.append("unsourced_claims")
            score *= 0.8
            confidence += fact_confidence * 0.25
        
        # 3. HEURISTIC: Response contradicts input
        contradiction_score = self._check_contradiction(prompt, response)
        if contradiction_score < 0.5:
            reasons.append("contradicts_prompt")
            score *= 0.6
            confidence += 0.4
        
        # 4. LLM-BASED: Fact checking (expensive, skip if already low score)
        if score > 0.5:
            llm_score, llm_confidence = self._llm_fact_check(
                prompt, response, analysis_model_id
            )
            score *= llm_score
            confidence += llm_confidence * 0.25
            if llm_score < 0.5:
                reasons.append("llm_fact_check_failed")
        
        # Normalize score
        final_score = min(1.0, max(0.0, score))
        final_confidence = min(1.0, confidence)
        
        # Flag response if score too low AND confidence high enough
        should_reject = final_score < 0.3 and final_confidence > 0.6
        
        return {
            "hallucination_score": final_score,
            "confidence": final_confidence,
            "reasons": reasons,
            "flag": should_reject
        }
    
    def _check_suspicious_patterns(self, response: str) -> tuple[float, float]:
        """Check for language patterns that suggest making stuff up."""
        matches = 0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                matches += 1
        
        # 0 matches: 1.0 (good), 1-2 matches: 0.7, 3+ matches: 0.3
        if matches == 0:
            return 1.0, 0.4  # Low confidence (pattern absence isn't proof)
        elif matches <= 2:
            return 0.7, 0.5
        else:
            return 0.3, 0.8
    
    def _check_unsourced_facts(self, prompt: str, response: str) -> tuple[float, float]:
        """
        Check if response makes factual claims without citing sources.
        Focus on claims that sound specific but have no context in prompt.
        """
        factual_keywords = [
            "fue", "ocurrió", "sucedió", "pasó", "murió", "nació",
            "en el año", "el día", "la fecha", "happened", "occurred"
        ]
        
        has_facts = any(kw in response.lower() for kw in factual_keywords)
        has_sources = any(
            phrase in response.lower() 
            for phrase in ["según", "fuente", "artículo", "estudio", "fuente:", "wikipedia"]
        )
        
        # Facts without sources = suspicious
        if has_facts and not has_sources:
            return 0.6, 0.7  # Moderate concern
        
        return 1.0, 0.3  # Either no facts or properly sourced
    
    def _check_contradiction(self, prompt: str, response: str) -> float:
        """Check if response directly contradicts the question."""
        # Naive: if prompt has negation and response affirms, or vice versa
        prompt_has_negation = any(
            word in prompt.lower() 
            for word in ["no es", "no hay", "no existe", "no puedo", "no sé"]
        )
        response_has_affirmation = any(
            word in response.lower()
            for word in ["es", "hay", "existe", "puedo", "sé"]
        )
        
        # If contradictory, lower score
        if prompt_has_negation and response_has_affirmation:
            return 0.5
        
        return 1.0
    
    def _llm_fact_check(
        self,
        prompt: str,
        response: str,
        analysis_model_id: str | None = None,
    ) -> tuple[float, float]:
        """Use LLM as fact-checker (expensive operation)."""
        fact_check_prompt = f"""
        Eres un fact-checker riguroso. Evalúa si la siguiente respuesta 
        es factualmente correcta o contiene alucinaciones.
        
        Pregunta del usuario: {str(prompt)[:300]}
        
        Respuesta a verificar: {str(response)[:500]}
        
        Responde ÚNICAMENTE con:
        VERIFICADA - la respuesta es correcta o razonable
        INCOMPLETA - la respuesta parece correcta pero le falta contexto
        CUESTIONABLE - la respuesta tiene hechos dudosos
        FALSA - la respuesta contiene alucinaciones claras
        
        Veredicto:"""
        
        try:
            result = self.models_handler.generate_text(
                fact_check_prompt,
                required_complexity=ModelComplexity.SMALL,
                model_id=analysis_model_id,
                max_new_tokens=3
            ).upper().strip()
            
            if "VERIFICADA" in result:
                return 1.0, 0.8
            elif "INCOMPLETA" in result:
                return 0.8, 0.7
            elif "CUESTIONABLE" in result:
                return 0.5, 0.8
            else:  # FALSA or unknown
                return 0.2, 0.8
        
        except Exception as e:
            print(f"[WARNING] Fact check failed: {e}")
            return 0.5, 0.2  # Neutral on error
```

---

## 2. Dynamic Alpha Adjustment

### Archivo: `src/infrastructure/repositories/router_stats_repository.py` (ADD METHOD)

```python
def get_adaptive_feedback_alpha(self, model_id: str) -> float:
    """
    Calculate adaptive alpha for exponential smoothing.
    
    Alpha decreases as:
    1. Sample count increases (more confidence in existing estimate)
    2. Variance decreases (less volatility = higher confidence)
    3. Age of data increases (old data = lower effective count)
    
    Returns: alpha in range [0.05, 0.5]
      - 0.05 = trust existing estimate heavily
      - 0.5 = treat new feedback equally
    """
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        
        # Get recent effectiveness scores (last 7 days)
        cursor.execute("""
            SELECT effectiveness_score, timestamp 
            FROM model_stats 
            WHERE model_id = ? AND timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 100
        """, (model_id,))
        
        rows = cursor.fetchall()
        
    if not rows:
        # No history, be open to feedback
        return 0.5
    
    scores = [float(row[0]) for row in rows]
    
    # 1. Sample count factor: more samples = lower alpha
    sample_count = len(rows)
    sample_factor = 1.0 / (1.0 + (sample_count / 20.0))  # Asymptotes at 0.05
    
    # 2. Variance factor: lower variance = lower alpha
    import numpy as np
    variance = float(np.var(scores))
    variance_factor = min(1.0, variance * 2.0)  # Variance 0-0.5 → factor 0-1
    
    # 3. Recency: recent data = higher alpha
    # (timestamp weights are implicit in query order, not used here for simplicity)
    
    # Combine factors: alpha = base * (1 - sample_factor) * (1 - variance_factor)
    base_alpha = 0.3
    adaptive_alpha = base_alpha * max(0.05, (sample_factor + variance_factor) / 2.0)
    
    print(f"[DEBUG] Model {model_id}: samples={sample_count}, "
          f"variance={variance:.3f}, alpha={adaptive_alpha:.3f}")
    
    return min(0.5, max(0.05, adaptive_alpha))


def update_effectiveness_with_adaptive_alpha(
    self,
    request_id: str,
    new_feedback_score: float
) -> tuple[float, float]:
    """
    Update effectiveness using adaptive alpha instead of fixed 0.35.
    
    Returns: (effectiveness_old, effectiveness_new)
    """
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        
        # Get current effectiveness
        cursor.execute("""
            SELECT effectiveness_score, model_id FROM model_stats
            WHERE request_id = ?
        """, (request_id,))
        
        row = cursor.fetchone()
        if not row:
            return 0.0, 0.0
        
        effectiveness_old = float(row[0])
        model_id = str(row[1])
        
        # Get adaptive alpha
        alpha = self.get_adaptive_feedback_alpha(model_id)
        
        # Update: E_new = (1 - alpha) * E_old + alpha * feedback
        effectiveness_new = (1.0 - alpha) * effectiveness_old + alpha * new_feedback_score
        
        # Save
        cursor.execute("""
            UPDATE model_stats 
            SET effectiveness_score = ? WHERE request_id = ?
        """, (effectiveness_new, request_id))
        
        conn.commit()
    
    return effectiveness_old, effectiveness_new
```

---

## 3. Temporal Drift Detection

### Archivo: `src/application/router/drift_detector.py` (NEW FILE)

```python
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Optional

class DriftDetector:
    """
    Monitors model performance over time and detects degradation.
    Implements automatic cooldown when drift is detected.
    """
    
    DRIFT_THRESHOLD = -0.2  # 20% drop = anomaly
    COOLDOWN_MINUTES = 15
    WINDOW_HOURS = 24
    
    def __init__(self, stats_repo) -> None:
        self.stats_repo = stats_repo
        self._model_cooldowns: Dict[str, datetime] = {}
    
    def check_and_flag_drift(self, model_id: str) -> Dict[str, any]:
        """
        Compares recent effectiveness vs 24h average.
        Returns:
        {
            "has_drift": bool,
            "drift_value": float (-0.5 = 50% worse),
            "recent_avg": float,
            "historical_avg": float,
            "should_cooldown": bool,
            "cooldown_until": datetime | None
        }
        """
        
        # Check if model is in cooldown
        if model_id in self._model_cooldowns:
            if datetime.now() < self._model_cooldowns[model_id]:
                return {
                    "has_drift": False,
                    "is_in_cooldown": True,
                    "cooldown_until": self._model_cooldowns[model_id]
                }
            else:
                del self._model_cooldowns[model_id]
        
        with sqlite3.connect(self.stats_repo.db_path) as conn:
            cursor = conn.cursor()
            
            # Last 2 hours (recent)
            cursor.execute("""
                SELECT AVG(effectiveness_score), COUNT(*) 
                FROM model_stats 
                WHERE model_id = ? AND timestamp > datetime('now', '-2 hours')
            """, (model_id,))
            recent = cursor.fetchone()
            recent_avg = float(recent[0]) if recent and recent[0] else 0.5
            recent_count = int(recent[1]) if recent else 0
            
            # Last 24 hours (historical)
            cursor.execute("""
                SELECT AVG(effectiveness_score), COUNT(*)
                FROM model_stats 
                WHERE model_id = ? AND timestamp > datetime('now', '-24 hours')
            """, (model_id,))
            historical = cursor.fetchone()
            historical_avg = float(historical[0]) if historical and historical[0] else 0.5
            historical_count = int(historical[1]) if historical else 0
        
        # Need enough samples to detect drift reliably
        if recent_count < 3:
            return {
                "has_drift": False,
                "insufficient_data": True,
                "recent_samples": recent_count
            }
        
        drift_value = recent_avg - historical_avg
        has_drift = drift_value < self.DRIFT_THRESHOLD
        
        if has_drift:
            # Enter cooldown
            cooldown_until = datetime.now() + timedelta(minutes=self.COOLDOWN_MINUTES)
            self._model_cooldowns[model_id] = cooldown_until
            
            print(f"[WARNING] Drift detected for {model_id}: "
                  f"{drift_value:.2f} (recent: {recent_avg:.2f}, historical: {historical_avg:.2f}). "
                  f"Entering cooldown until {cooldown_until}")
            
            return {
                "has_drift": True,
                "drift_value": drift_value,
                "recent_avg": recent_avg,
                "historical_avg": historical_avg,
                "should_cooldown": True,
                "cooldown_until": cooldown_until
            }
        
        return {
            "has_drift": False,
            "drift_value": drift_value,
            "recent_avg": recent_avg,
            "historical_avg": historical_avg,
            "should_cooldown": False
        }
    
    def get_cooldown_status(self, model_id: str) -> Optional[datetime]:
        """Returns when model can be used again, or None if available."""
        if model_id not in self._model_cooldowns:
            return None
        
        cooldown_time = self._model_cooldowns[model_id]
        if datetime.now() >= cooldown_time:
            del self._model_cooldowns[model_id]
            return None
        
        return cooldown_time
```

---

## 4. Integración en RouterService

### Modificación: `src/application/router/router_service.py`

```python
# Add imports at top
from application.router.hallucination_detector import HallucinationDetector
from application.router.drift_detector import DriftDetector

# In __init__
def __init__(self, stats_repo: RouterStatsRepository, embedding_engine: EmbeddingEngine) -> None:
    # ... existing code ...
    self.hallucination_detector = HallucinationDetector(models_handler)  # Need to pass ModelsHandler
    self.drift_detector = DriftDetector(stats_repo)

# In route_request(), after _select_best_model():
def route_request(self, prompt: str, session_id: str, embedding: list[float]) -> str:
    # ... existing k-NN logic ...
    
    best_model = self._select_best_model(similar_performance)
    
    # NEW: Check drift status
    drift_status = self.drift_detector.check_and_flag_drift(best_model.huggingface_id)
    if drift_status.get("should_cooldown"):
        print(f"[INFO] Model {best_model.huggingface_id} in drift cooldown. Finding alternative...")
        best_model = self._select_best_model(similar_performance, exclude_model=best_model.huggingface_id)
    
    return best_model.huggingface_id
```

---

## Métricas Prometheus (Agregar)

```python
# En infrastructure/observability/metrics.py

from prometheus_client import Counter, Gauge, Histogram

HALLUCINATION_DETECTIONS = Counter(
    'hallucination_detections_total',
    'Total hallucinations detected',
    ['model_id', 'severity']  # severity: low, medium, high
)

DRIFT_ALERTS = Counter(
    'drift_alerts_total',
    'Total model drift alerts',
    ['model_id']
)

MODEL_QUALITY_DRIFT = Gauge(
    'model_quality_drift',
    'Recent vs historical effectiveness delta',
    ['model_id']
)

FEEDBACK_ALPHA_ADAPTIVE = Gauge(
    'feedback_alpha_adaptive',
    'Current adaptive alpha value',
    ['model_id']
)
```

---

## Testing

```python
# tests/test_hallucination_detector.py

import pytest
from application.router.hallucination_detector import HallucinationDetector

@pytest.fixture
def detector():
    # Mock ModelsHandler
    return HallucinationDetector(mock_models_handler)

def test_detects_suspicious_patterns():
    response = "Según invento yo, el presidente en 1920 fue Mario."
    result = detector.detect_hallucinations("¿Quién fue presidente?", response)
    assert result["hallucination_score"] < 0.7
    assert "suspicious_pattern_language" in result["reasons"]

def test_detects_unsourced_facts():
    response = "Albert Einstein nació en 1960 en Nueva York."
    result = detector.detect_hallucinations("¿Cuándo nació Einstein?", response)
    assert result["hallucination_score"] < 0.8

def test_allows_sourced_claims():
    response = "Según Wikipedia, Einstein nació en 1879 en Alemania."
    result = detector.detect_hallucinations("¿Cuándo nació Einstein?", response)
    assert result["hallucination_score"] > 0.8
```

