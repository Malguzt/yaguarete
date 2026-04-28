import re
from typing import Any

from infrastructure.transformers_engine.model_catalog import ModelComplexity
from infrastructure.transformers_engine.models_handler import ModelsHandler


class HallucinationDetector:
    """
    Lightweight hallucination detector:
    - fast heuristics first
    - optional LLM fact-check as second pass
    """

    def __init__(self, models_handler: ModelsHandler) -> None:
        self.models_handler = models_handler
        self.suspicious_patterns = [
            r"(seg[uú]n|afirma que|dice que|menciona que)\s+[^.]{20,}",
            r"\b(invento|imagino|supongo)\b",
            r"\b(no existe|nunca ocurri[oó]|falso)\b",
        ]

    def detect_hallucinations(
        self,
        prompt: str,
        response: str,
        analysis_model_id: str | None = None,
        use_llm_fact_check: bool = True,
    ) -> dict[str, Any]:
        score = 1.0
        confidence = 0.0
        reasons: list[str] = []

        pattern_score, pattern_confidence = self._check_suspicious_patterns(response)
        if pattern_score < 0.7:
            score *= pattern_score
            confidence += pattern_confidence * 0.35
            reasons.append("suspicious_patterns")

        unsourced_score, unsourced_confidence = self._check_unsourced_facts(prompt, response)
        if unsourced_score < 0.9:
            score *= unsourced_score
            confidence += unsourced_confidence * 0.25
            reasons.append("unsourced_claims")

        contradiction_score = self._check_contradiction(prompt, response)
        if contradiction_score < 1.0:
            score *= contradiction_score
            confidence += 0.35
            reasons.append("prompt_contradiction")

        if use_llm_fact_check and score > 0.45:
            llm_score, llm_confidence = self._llm_fact_check(prompt, response, analysis_model_id=analysis_model_id)
            score *= llm_score
            confidence += llm_confidence * 0.25
            if llm_score < 0.7:
                reasons.append("llm_fact_check")

        final_score = min(1.0, max(0.0, score))
        final_confidence = min(1.0, max(0.0, confidence))
        return {
            "hallucination_score": final_score,
            "confidence": final_confidence,
            "reasons": reasons,
            "flag": final_score < 0.35 and final_confidence > 0.6,
        }

    def _check_suspicious_patterns(self, response: str) -> tuple[float, float]:
        matches = 0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                matches += 1
        if matches == 0:
            return 1.0, 0.3
        if matches <= 2:
            return 0.75, 0.55
        return 0.45, 0.8

    def _check_unsourced_facts(self, prompt: str, response: str) -> tuple[float, float]:
        factual_tokens = [
            "en el año",
            "ocurrió",
            "sucedió",
            "fue",
            "happened",
            "occurred",
            "born",
            "died",
        ]
        source_tokens = ["según", "fuente", "source", "wikipedia", "paper", "artículo", "estudio"]

        has_factual_claim = any(token in response.lower() for token in factual_tokens)
        prompt_requires_facts = any(token in prompt.lower() for token in ("quién", "cuándo", "fecha", "historia", "año"))
        has_source = any(token in response.lower() for token in source_tokens)

        if has_factual_claim and prompt_requires_facts and not has_source:
            return 0.8, 0.7
        return 1.0, 0.3

    def _check_contradiction(self, prompt: str, response: str) -> float:
        prompt_has_negation = any(token in prompt.lower() for token in ("no ", "nunca", "jamás", "cannot", "can't"))
        response_has_positive_assertion = any(
            token in response.lower() for token in ("sí", "si ", "correcto", "es cierto", "yes", "true")
        )
        if prompt_has_negation and response_has_positive_assertion:
            return 0.7
        return 1.0

    def _llm_fact_check(
        self,
        prompt: str,
        response: str,
        analysis_model_id: str | None = None,
    ) -> tuple[float, float]:
        check_prompt = (
            "Eres un fact-checker estricto. Evalúa la respuesta y responde SOLO con una palabra:\n"
            "VERIFICADA, INCOMPLETA, CUESTIONABLE o FALSA.\n\n"
            f"Pregunta: {prompt[:350]}\n"
            f"Respuesta: {response[:550]}\n"
            "Veredicto:"
        )
        try:
            result = self.models_handler.generate_text(
                check_prompt,
                required_complexity=ModelComplexity.SMALL,
                model_id=analysis_model_id,
                max_new_tokens=5,
                temperature=0.1,
            ).upper()
            if "VERIFICADA" in result:
                return 1.0, 0.8
            if "INCOMPLETA" in result:
                return 0.85, 0.7
            if "CUESTIONABLE" in result:
                return 0.65, 0.75
            if "FALSA" in result:
                return 0.25, 0.85
            return 0.75, 0.35
        except Exception as e:
            print(f"[WARNING] Hallucination LLM fact-check failed: {e}")
            return 0.8, 0.2
