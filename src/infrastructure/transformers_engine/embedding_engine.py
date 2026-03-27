import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List

class EmbeddingEngine:
    """
    Local engine for generating embeddings using a small, efficient model.
    Used for topic detection and vector similarity.
    """
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        print(f"[INFO] Embedding engine initialized on {self.device}")

    def get_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()[0].tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
