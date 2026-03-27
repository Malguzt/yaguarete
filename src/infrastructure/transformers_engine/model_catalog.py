from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ModelComplexity(Enum):
    SMALL = "small"   # Fits in single GPU easily, fast inference (e.g. <3B params)
    MEDIUM = "medium" # Might need multiple GPUs or GPU+RAM (e.g. 7B - 14B)
    LARGE = "large"   # Requires GPU+RAM offloading (e.g. 32B+)

class ModelSpecialty(Enum):
    GENERAL = "general"
    CODE = "code"
    CHAT = "chat"
    REASONING = "reasoning"

@dataclass
class ModelDefinition:
    huggingface_id: str
    complexity: ModelComplexity
    specialty: ModelSpecialty
    # Estimated VRAM required in GB for full GPU load (FP16 typical)
    estimated_vram_gb: float
    cost_per_1k_chars: float = 0.0001 # Default base cost

class ModelCatalog:
    """Catalog of locally available or configured Hugging Face models."""
    
    def __init__(self):
        # Define the available models in our local environment
        self.models: List[ModelDefinition] = [
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=4.0,
                cost_per_1k_chars=0.00005
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-7B-Instruct",
                complexity=ModelComplexity.MEDIUM,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=15.0,
                cost_per_1k_chars=0.0002
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-Coder-1.5B",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CODE,
                estimated_vram_gb=4.0,
                cost_per_1k_chars=0.00005
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-Coder-7B",
                complexity=ModelComplexity.MEDIUM,
                specialty=ModelSpecialty.CODE,
                estimated_vram_gb=15.0,
                cost_per_1k_chars=0.0002
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-32B-Instruct",
                complexity=ModelComplexity.LARGE,
                specialty=ModelSpecialty.REASONING,
                estimated_vram_gb=68.0,
                cost_per_1k_chars=0.001
            ),
            ModelDefinition(
                huggingface_id="distilbert-base-uncased-finetuned-sst-2-english",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT, # Used for sentiment
                estimated_vram_gb=1.0,
                cost_per_1k_chars=0.00001
            )
        ]
        
    def find_best_model(self, required_complexity: ModelComplexity, required_specialty: ModelSpecialty) -> Optional[ModelDefinition]:
        """Finds the best matching model based on complexity and specialty."""
        
        # 1. Exact match
        for model in self.models:
            if model.complexity == required_complexity and model.specialty == required_specialty:
                return model
                
        # 2. Match complexity, fallback to general/chat
        for model in self.models:
            if model.complexity == required_complexity and model.specialty in [ModelSpecialty.GENERAL, ModelSpecialty.CHAT]:
                return model
                
        # 3. Match specialty, allow higher complexity if small was requested and not found,
        # or allow lower complexity if large was requested and not found.
        # Simple fallback for now: return first matching specialty
        for model in self.models:
            if model.specialty == required_specialty:
                return model
                
        # 4. Ultimate fallback: First available chat/general model
        for model in self.models:
            if model.specialty in [ModelSpecialty.GENERAL, ModelSpecialty.CHAT]:
                return model
                
        # If catalog is totally empty (shouldn't happen)
        return self.models[0]

    def get_default_model(self) -> ModelDefinition:
        """Returns a safe default model."""
        return self.find_best_model(ModelComplexity.SMALL, ModelSpecialty.CHAT)
