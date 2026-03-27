from typing import Dict
from infrastructure.transformers_engine.models_handler import ModelsHandler
from infrastructure.transformers_engine.model_catalog import ModelComplexity

class CognitivePlanner:
    """
    Uses a small local model to analyze the cognitive load of a request.
    Decides if the task is Simple, Medium, or Complex.
    """
    def __init__(self, models_handler: ModelsHandler):
        self.models_handler = models_handler
        # Planning is always done by the smallest efficient model
        self.planning_model_complexity = ModelComplexity.SMALL

    def estimate_load(self, prompt: str) -> ModelComplexity:
        planning_prompt = f"""
        Analiza el siguiente requerimiento del usuario y determina su complejidad cognitiva.
        Responde ÚNICAMENTE con una de estas tres palabras: SMALL, MEDIUM, LARGE.
        
        - SMALL: Preguntas fácticas simples, saludos, comandos directos.
        - MEDIUM: Análisis de textos, resúmenes, explicaciones detalladas, escritura creativa simple.
        - LARGE: Razonamiento lógico complejo, planificación de múltiples pasos, depuración de código difícil, síntesis de múltiples conceptos.
        
        Requerimiento: {prompt}
        Complejidad:"""
        
        try:
            # We use the models_handler to run inference on the small model
            result = self.models_handler.generate_text(
                planning_prompt, 
                required_complexity=self.planning_model_complexity,
                max_new_tokens=10
            )
            
            result = result.upper().strip()
            if "LARGE" in result: return ModelComplexity.LARGE
            if "MEDIUM" in result: return ModelComplexity.MEDIUM
            return ModelComplexity.SMALL
        except Exception as e:
            print(f"[ERROR] Planning failed: {e}. Falling back to SMALL.")
            return ModelComplexity.SMALL
