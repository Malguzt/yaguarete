import uuid
import time
from typing import List, Dict, Optional
from infrastructure.transformers_engine.model_catalog import ModelCatalog, ModelComplexity, ModelSpecialty
from infrastructure.repositories.router_stats_repository import RouterStatsRepository
from infrastructure.transformers_engine.embedding_engine import EmbeddingEngine
from application.router.cognitive_planner import CognitivePlanner

class RouterService:
    """
    Advanced router that selects the best model for a given request.
    Considers: topic, stats, cognitive load, and session history.
    """
    def __init__(self, stats_repo: RouterStatsRepository, embedding_engine: EmbeddingEngine):
        self.catalog = ModelCatalog()
        self.stats_repo = stats_repo
        self.embedding_engine = embedding_engine

    def route_request(self, prompt: str, session_id: str, embedding: List[float]) -> str:
        # 1. Similarity-Based Performance Analysis (k-NN)
        # Find how models performed on semantically similar requests in the past
        similar_performance = self.stats_repo.get_similar_performance(embedding)
        
        # 2. Check session history for immediate feedback (repreguntas similares)
        last_input = self.stats_repo.get_last_session_input(session_id)
        if last_input:
            similarity = self.embedding_engine.calculate_similarity(embedding, last_input["embedding"])
            if similarity > 0.85:
                print(f"[INFO] High similarity ({similarity:.2f}) detected in session {session_id}. Penalizing previous model.")
                self.stats_repo.penalize_last_request(session_id)

        # 3. Dynamic Selection based on Similar + Global stats
        best_model = self._select_best_model(similar_performance)
        
        # 4. Log session input for future distance checks
        self.stats_repo.log_session_input(session_id, prompt, embedding)
        
        return best_model.huggingface_id

    def _select_best_model(self, similar_stats: Dict[str, Dict]):
        all_models = self.catalog.models
        
        scores = {}
        for model in all_models:
            mid = model.huggingface_id
            global_stats = self.stats_repo.get_model_performance(mid)
            local_stats = similar_stats.get(mid)
            
            # Base data from catalog
            base_cost = model.cost_per_1k_chars
            
            # 1. Calculate weighted effectiveness
            # Priority: Similarity Stats (local) > Global Stats > Default (1.0)
            if local_stats:
                eff_base = local_stats["avg_effectiveness"]
                eff_format = local_stats["avg_format"]
                eff_judge = local_stats["avg_judge"]
                eff_sentiment = (local_stats["avg_sentiment"] + 1.0) / 2.0
                confidence = 0.8 # Higher weight to local similarity
            elif global_stats:
                eff_base = global_stats["avg_effectiveness"]
                eff_format = global_stats["avg_format"]
                eff_judge = global_stats["avg_judge"]
                eff_sentiment = (global_stats["avg_sentiment"] + 1.0) / 2.0
                confidence = 0.4
            else:
                eff_base = 1.0
                eff_format = 1.0
                eff_judge = 1.0
                eff_sentiment = 0.5
                confidence = 0.1

            eff = (eff_base * 0.4) + (eff_format * 0.2) + (eff_judge * 0.3) + (eff_sentiment * 0.1)
            
            # 2. Get Duration & Cost
            dur = 2000.0 # Default
            cost = base_cost
            
            if local_stats:
                dur = local_stats["avg_duration"]
                cost = local_stats["avg_cost"]
            elif global_stats:
                dur = global_stats["avg_duration"]
                cost = global_stats["avg_cost"]
            
            # 3. Final Scoring Formula
            # Score = (Effectiveness / (Cost * Duration))
            # We add a small constant to cost/dur to avoid division by zero
            score = (eff * 1000000.0) / (max(cost, 0.00001) * max(dur, 100.0))
            
            # Apply confidence boost if we have high similarity local stats
            if local_stats:
                score *= 1.5
                
            scores[mid] = score
            print(f"[DEBUG] Model {mid} score: {score:.2f} (local: {local_stats is not None})")

        if not scores:
            return self.catalog.get_default_model()
            
        best_mid = max(scores.items(), key=lambda x: x[1])[0]
        for m in all_models:
            if m.huggingface_id == best_mid:
                return m
        return self.catalog.get_default_model()

    def select_shadow_model(self, primary_model_id: str, complexity: ModelComplexity) -> Optional[str]:
        """Randomly selects a second model of the same complexity tier for comparison."""
        import random
        if random.random() > 0.95: # 5% chance of shadowing
            candidates = [m for m in self.catalog.models if m.complexity == complexity and m.huggingface_id != primary_model_id]
            if candidates:
                return random.choice(candidates).huggingface_id
        return None
