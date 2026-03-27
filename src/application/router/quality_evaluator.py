import json
import re
from typing import Dict
from infrastructure.transformers_engine.models_handler import ModelsHandler
from infrastructure.transformers_engine.model_catalog import ModelComplexity

class QualityEvaluator:
    """
    Evaluates the quality of a model's response using local heuristics and models.
    """
    def __init__(self, models_handler: ModelsHandler):
        self.models_handler = models_handler

    def evaluate_response(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Returns a dict of scores between 0.0 and 1.0
        """
        return {
            "format_score": self._check_format(prompt, response),
            "density_score": self._check_density(prompt, response),
            "judge_score": self._judge_relevance(prompt, response),
            "sentiment_score": self._check_sentiment(response)
        }

    def _check_sentiment(self, text: str) -> float:
        """Analyzes sentiment of the response locally."""
        sentiment_prompt = f"""
        Analyze the sentiment of the following text. 
        Respond with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE.
        
        Text: {str(text)[:500]}
        Sentiment:"""
        
        try:
            result = self.models_handler.generate_text(
                sentiment_prompt,
                required_complexity=ModelComplexity.SMALL,
                max_new_tokens=5
            ).upper()
            
            if "POSITIVE" in result: return 1.0
            if "NEGATIVE" in result: return -1.0
            return 0.0
        except:
            return 0.5 # Neutral fallback

    def _check_format(self, prompt: str, response: str) -> float:
        # If prompt asks for JSON, check if response is valid JSON
        if "json" in prompt.lower():
            try:
                # Find content between { } or [ ]
                json_match = re.search(r"(\{.*\}|\[.*\])", response, re.DOTALL)
                if json_match:
                    json.loads(json_match.group(1))
                    return 1.0
                return 0.0
            except:
                return 0.0
        
        # If prompt asks for code, check for code blocks
        if any(kw in prompt.lower() for kw in ["código", "python", "javascript", "code"]):
            if "```" in response:
                return 1.0
            return 0.5 # Partial credit if code is there but not in blocks
            
        return 1.0 # Default if no specific format requested

    def _check_density(self, prompt: str, response: str) -> float:
        # Heuristic: ratio of response length to prompt length
        # For complex prompts, very short responses are suspicious
        input_len = len(prompt)
        output_len = len(response)
        
        if input_len > 1000 and output_len < 100:
            return 0.2
        if input_len > 500 and output_len < 50:
            return 0.5
        return 1.0

    def _judge_relevance(self, prompt: str, response: str) -> float:
        """Uses the small local model to judge if the response is relevant."""
        judge_prompt = f"""
        Actúa como un juez de calidad. Determina si la respuesta satisface el pedido del usuario.
        Responde ÚNICAMENTE con 'SÍ' o 'NO'.
        
        Pedido: {str(prompt)[:500]}
        Respuesta: {str(response)[:500]}
        
        ¿Es satisfactoria? (SÍ/NO):"""
        
        try:
            result = self.models_handler.generate_text(
                judge_prompt, 
                required_complexity=ModelComplexity.SMALL,
                max_new_tokens=5
            )
            if "SÍ" in result.upper() or "SI" in result.upper():
                return 1.0
            return 0.0
        except:
            return 1.0 # Default to neutral if judge fails
