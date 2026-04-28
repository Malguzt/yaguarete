import os
from typing import Optional
import torch
from .model_catalog import ModelCatalog, ModelComplexity, ModelSpecialty, ModelDefinition, ModelProvider

class ModelRouter:
    """Decides which model to use based on prompt characteristics and cost-effectiveness."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self.catalog = catalog
        self.default_complexity = self._read_default_complexity()
        self.prefer_local = os.getenv("PREFER_LOCAL_MODELS", "true").lower() in ("true", "1", "yes")

    def _read_default_complexity(self) -> ModelComplexity:
        raw = os.getenv("DEFAULT_BOT_COMPLEXITY", "medium").strip().lower()
        mapping = {
            "small": ModelComplexity.SMALL,
            "medium": ModelComplexity.MEDIUM,
            "large": ModelComplexity.LARGE,
        }
        return mapping.get(raw, ModelComplexity.MEDIUM)

    def route_prompt(self, prompt: str, required_complexity: Optional[ModelComplexity] = None) -> ModelDefinition:
        """
        Analyzes the prompt and returns the best model definition.
        Prioritizes: effectiveness (quality) with minimum cost.
        
        Strategy:
        - Try local models first (free, private)
        - Fall back to free/cheap remote models (Llama 3.1 via OpenRouter)
        - Use premium remote models only if needed for complexity/reasoning
        """
        # Determine Specialty
        specialty = self._determine_specialty(prompt)
        
        # Determine Complexity (if not explicitly requested)
        if required_complexity is None:
            required_complexity = self._determine_complexity(prompt, specialty)

        # Cost-optimized model selection
        model_def = self._select_cost_optimal_model(required_complexity, specialty)
        
        if model_def is None:
            model_def = self.catalog.get_default_model()
            
        return model_def

    def _select_cost_optimal_model(self, complexity: ModelComplexity, specialty: ModelSpecialty) -> Optional[ModelDefinition]:
        """
        Select the best model balancing quality and cost.
        
        Priority order:
        1. Local model (if available and prefer_local=true)
        2. Free remote model (Llama 3.1)
        3. Cheap remote model (gpt-4o-mini for MEDIUM)
        4. Premium remote model (Claude 3.5 for LARGE)
        """
        
        # 1. Try to find local model first
        if self.prefer_local:
            local_model = self.catalog.find_best_local_model(complexity, specialty)
            if local_model:
                return local_model
        
        # 2. If no local model or prefer_local=false, try remote models in cost order
        remote_candidates = []
        
        # Collect free/ultra-cheap models
        for model in self.catalog.get_all_remote_models():
            if model.cost_per_1k_chars < 0.000001:
            # Free tier
                if model.complexity == complexity or model.complexity == ModelComplexity.SMALL:
                    remote_candidates.append((model, 0))
                    # Priority 0 = free
        
        # Collect cheap models
        if complexity == ModelComplexity.SMALL or complexity == ModelComplexity.MEDIUM:
            for model in self.catalog.get_all_remote_models():
                if model.cost_per_1k_chars < 0.0001:
                # < $0.0001/1k chars
                    if model.complexity == complexity:
                        remote_candidates.append((model, 1))
                        # Priority 1 = cheap
        
        # Collect exact match models
        for model in self.catalog.get_all_remote_models():
            if model.complexity == complexity and model.specialty == specialty:
                priority = 2 if model.cost_per_1k_chars > 0.001 else 1
                remote_candidates.append((model, priority))
        
        # Sort by priority (lower is better) then by cost
        if remote_candidates:
            remote_candidates.sort(key=lambda x: (x[1], x[0].cost_per_1k_chars))
            return remote_candidates[0][0]
        
        # 3. Ultimate fallback: cheapest remote model overall
        cheapest_remote = min(
            self.catalog.get_all_remote_models(),
            key=lambda m: m.cost_per_1k_chars,
            default=None
        )
        if cheapest_remote:
            return cheapest_remote
        
        # 4. Fallback to default
        return self.catalog.get_default_model()

    def _determine_specialty(self, prompt: str) -> ModelSpecialty:
        """Simple heuristic to determine if it's a coding or general question."""
        code_keywords = ["python", "code", "código", "function", "función", "bash", "html", "css", "javascript", "bug", "error", "refactor"]
        lower_prompt = prompt.lower()
        
        if any(keyword in lower_prompt for keyword in code_keywords):
            return ModelSpecialty.CODE
            
        reasoning_keywords = ["analyze", "analiza", "evaluate", "evalúa", "compare", "compara", "plan", "architecture", "arquitectura", "solve", "resuelve", "why", "por qué"]
        if any(keyword in lower_prompt for keyword in reasoning_keywords):
            return ModelSpecialty.REASONING
            
        return ModelSpecialty.CHAT

    def _determine_complexity(self, prompt: str, specialty: ModelSpecialty) -> ModelComplexity:
        """
        Determine how complex the model needs to be.
        Complex reasoning or very long prompts need larger models.
        """
        if specialty == ModelSpecialty.REASONING:
            return ModelComplexity.LARGE

        # If no GPU is available, stay small by default.
        if not torch.cuda.is_available():
            if len(prompt) > 1000:
                return ModelComplexity.MEDIUM
            return ModelComplexity.SMALL

        # On GPU, default to medium to better utilize VRAM and quality.
        # Extremely short prompts can stay small.
        if len(prompt.strip()) <= 25:
            return ModelComplexity.SMALL

        if len(prompt) > 1600:
            return ModelComplexity.LARGE

        return self.default_complexity
