from dataclasses import dataclass
from enum import Enum
import json
import os
import time
from typing import Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from transformers import AutoConfig


class ModelComplexity(Enum):
    SMALL = "small"
    # Fits in single GPU easily, fast inference (e.g. <3B params)
    MEDIUM = "medium"
    # Might need quantization/offload for 7B+ on limited VRAM
    LARGE = "large"
    # Requires substantial offload / remote fallback


class ModelSpecialty(Enum):
    GENERAL = "general"
    CODE = "code"
    CHAT = "chat"
    REASONING = "reasoning"


class ModelProvider(Enum):
    LOCAL = "local"
    OPENROUTER = "openrouter"


@dataclass
class ModelDefinition:
    huggingface_id: str
    complexity: ModelComplexity
    specialty: ModelSpecialty
    # Estimated VRAM required in GB for full GPU load (FP16 typical)
    estimated_vram_gb: float
    cost_per_1k_chars: float = 0.0001
    # Default base cost
    provider: ModelProvider = ModelProvider.LOCAL
    # Where the model is hosted
    is_remote: bool = False
    # For convenience: True if provider != LOCAL
    supports_generation: bool = True
    # Normalized 0..1 signal used by routing (higher is better).
    popularity_score: float = 0.5


class ModelCatalog:
    """Catalog of locally available and remote (OpenRouter) models."""
    _openrouter_cache: list[ModelDefinition] = []
    _openrouter_cache_expires_at: float = 0.0
    _llmfit_local_cache: list[ModelDefinition] = []
    _llmfit_local_cache_expires_at: float = 0.0
    _last_llmfit_local_fallback: list[ModelDefinition] = []
    _llmfit_loadability_cache: dict[str, bool] = {}
    _llmfit_snapshot_path = os.getenv(
        "LLMFIT_SNAPSHOT_PATH",
        "data/llmfit_local_catalog_snapshot.json",
    )

    def __init__(self) -> None:
        # Local models tuned for constrained VRAM (8GB class) with scalable fallbacks.
        self.models: list[ModelDefinition] = [
            # --- LOCAL MODELS (SMALL) ---
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=4.0,
                cost_per_1k_chars=0.00004,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CODE,
                estimated_vram_gb=4.0,
                cost_per_1k_chars=0.00004,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-3B-Instruct",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=6.0,
                cost_per_1k_chars=0.00005,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-Coder-3B-Instruct",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CODE,
                estimated_vram_gb=6.0,
                cost_per_1k_chars=0.00005,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),
            ModelDefinition(
                huggingface_id="distilbert-base-uncased-finetuned-sst-2-english",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=1.0,
                cost_per_1k_chars=0.00001,
                provider=ModelProvider.LOCAL,
                is_remote=False,
                supports_generation=False,
            ),

            # --- LOCAL MODELS (MEDIUM) ---
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-7B-Instruct",
                complexity=ModelComplexity.MEDIUM,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=15.0,
                cost_per_1k_chars=0.0002,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                complexity=ModelComplexity.MEDIUM,
                specialty=ModelSpecialty.CODE,
                estimated_vram_gb=15.0,
                cost_per_1k_chars=0.0002,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),

            # --- LOCAL MODELS (LARGE) ---
            ModelDefinition(
                huggingface_id="Qwen/Qwen2.5-14B-Instruct",
                complexity=ModelComplexity.LARGE,
                specialty=ModelSpecialty.REASONING,
                estimated_vram_gb=30.0,
                cost_per_1k_chars=0.0006,
                provider=ModelProvider.LOCAL,
                is_remote=False,
            ),

            # --- OPENROUTER MODELS (SMALL) ---
            ModelDefinition(
                huggingface_id="meta-llama/llama-3.1-8b-instruct:free",
                complexity=ModelComplexity.SMALL,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=0.0,
                cost_per_1k_chars=0.000001,
                provider=ModelProvider.OPENROUTER,
                is_remote=True,
            ),

            # --- OPENROUTER MODELS (MEDIUM) ---
            ModelDefinition(
                huggingface_id="openai/gpt-4o-mini",
                complexity=ModelComplexity.MEDIUM,
                specialty=ModelSpecialty.CHAT,
                estimated_vram_gb=0.0,
                cost_per_1k_chars=0.000075,
                provider=ModelProvider.OPENROUTER,
                is_remote=True,
            ),

            # --- OPENROUTER MODELS (LARGE) ---
            ModelDefinition(
                huggingface_id="anthropic/claude-3.5-sonnet",
                complexity=ModelComplexity.LARGE,
                specialty=ModelSpecialty.REASONING,
                estimated_vram_gb=0.0,
                cost_per_1k_chars=0.00375,
                provider=ModelProvider.OPENROUTER,
                is_remote=True,
            ),
            ModelDefinition(
                huggingface_id="openai/gpt-4-turbo",
                complexity=ModelComplexity.LARGE,
                specialty=ModelSpecialty.REASONING,
                estimated_vram_gb=0.0,
                cost_per_1k_chars=0.002,
                provider=ModelProvider.OPENROUTER,
                is_remote=True,
            ),
        ]

        self._replace_local_models_from_llmfit_if_enabled()
        self._inject_dynamic_openrouter_models_if_enabled()
        self._apply_model_overrides_from_env()
        self._prioritize_by_llmfit_if_enabled()

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _is_llmfit_model_loadable(self, model_id: str) -> bool:
        cached = ModelCatalog._llmfit_loadability_cache.get(model_id)
        if cached is not None:
            return cached

        validate = os.getenv("LLMFIT_VALIDATE_MODEL_LOADABILITY", "1").lower() in ("1", "true", "yes")
        if not validate:
            ModelCatalog._llmfit_loadability_cache[model_id] = True
            return True

        trust_all = os.getenv("TRUST_REMOTE_CODE", "0") == "1"
        trust_allowlist = {
            m.strip() for m in os.getenv("TRUST_REMOTE_CODE_MODELS", "").split(",") if m.strip()
        }
        trust_for_this_model = trust_all or (model_id in trust_allowlist)

        try:
            AutoConfig.from_pretrained(model_id, trust_remote_code=trust_for_this_model)
            ModelCatalog._llmfit_loadability_cache[model_id] = True
            return True
        except Exception as e:
            text = str(e).lower()
            deterministic_blockers = (
                "gated repo",
                "unauthorized",
                "restricted",
                "trust_remote_code=true",
                "contains custom code",
                "requires you to execute the configuration file",
                "does not recognize this architecture",
                "does not recognize this model type",
            )
            if any(marker in text for marker in deterministic_blockers):
                print(f"[WARNING] Skipping incompatible LLMFit model {model_id}: {e}")
                ModelCatalog._llmfit_loadability_cache[model_id] = False
                return False

            # For transient failures, keep it eligible.
            ModelCatalog._llmfit_loadability_cache[model_id] = True
            return True

    def _inject_dynamic_openrouter_models_if_enabled(self) -> None:
        enabled = os.getenv("OPENROUTER_DYNAMIC_MODELS", "true").lower() in ("1", "true", "yes")
        if not enabled or not self._openrouter_enabled():
            return

        dynamic_models = self._get_dynamic_openrouter_models()
        if not dynamic_models:
            return

        self.models = [m for m in self.models if not m.is_remote]
        self.models.extend(dynamic_models)
        print(f"[INFO] Loaded {len(dynamic_models)} dynamic OpenRouter models")

    def _get_dynamic_openrouter_models(self) -> list[ModelDefinition]:
        now = time.time()
        if now < ModelCatalog._openrouter_cache_expires_at and ModelCatalog._openrouter_cache:
            return ModelCatalog._openrouter_cache

        ttl = int(os.getenv("OPENROUTER_DYNAMIC_MODELS_CACHE_TTL_SEC", "900"))
        if ttl < 30:
            ttl = 30
        fetched_models = self._fetch_openrouter_models()
        if fetched_models:
            ModelCatalog._openrouter_cache = fetched_models
            ModelCatalog._openrouter_cache_expires_at = now + ttl
        return ModelCatalog._openrouter_cache

    def _fetch_openrouter_models(self) -> list[ModelDefinition]:
        endpoint = os.getenv("OPENROUTER_MODELS_ENDPOINT", "https://openrouter.ai/api/v1/models").strip()
        if not endpoint:
            return []

        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib_request.Request(endpoint, headers=headers)
        try:
            with urllib_request.urlopen(request, timeout=6.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as err:
            print(f"[WARNING] Failed to fetch dynamic OpenRouter models from {endpoint}: {err}")
            return []

        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list) or not rows:
            return []

        limit = int(os.getenv("OPENROUTER_DYNAMIC_MODELS_LIMIT", "30"))
        limit = max(5, min(limit, 200))
        price_weight = self._safe_float(os.getenv("OPENROUTER_DYNAMIC_WEIGHT_PRICE", "0.65"), 0.65)
        popularity_weight = self._safe_float(os.getenv("OPENROUTER_DYNAMIC_WEIGHT_POPULARITY", "0.35"), 0.35)
        total_weight = max(0.0001, price_weight + popularity_weight)
        price_weight = price_weight / total_weight
        popularity_weight = popularity_weight / total_weight

        candidates: list[dict[str, object]] = []
        total_rows = len(rows)
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id", "")).strip()
            if not model_id:
                continue
            if not self._supports_text_generation(row, model_id):
                continue
            price = self._price_per_1k_chars(row)
            popularity = self._extract_popularity(row, idx, total_rows)
            complexity = self._infer_complexity(row, model_id)
            specialty = self._infer_specialty(row, model_id)
            candidates.append({
                "id": model_id,
                "price": price,
                "popularity": popularity,
                "complexity": complexity,
                "specialty": specialty,
            })

        if not candidates:
            return []

        prices = [float(c["price"]) for c in candidates]
        min_price = min(prices)
        max_price = max(prices)
        price_span = max(max_price - min_price, 1e-9)

        def _price_score(value: float) -> float:
            return 1.0 - ((value - min_price) / price_span)

        for c in candidates:
            popularity = float(c["popularity"])
            price_score = _price_score(float(c["price"]))
            c["selection_score"] = (price_weight * price_score) + (popularity_weight * popularity)

        ranked = sorted(
            candidates,
            key=lambda c: (
                float(c["selection_score"]),
                float(c["popularity"]),
                -float(c["price"]),
            ),
            reverse=True,
        )

        selected: list[ModelDefinition] = []
        for c in ranked[:limit]:
            selected.append(
                ModelDefinition(
                    huggingface_id=str(c["id"]),
                    complexity=c["complexity"],
                    specialty=c["specialty"],
                    estimated_vram_gb=0.0,
                    cost_per_1k_chars=float(c["price"]),
                    provider=ModelProvider.OPENROUTER,
                    is_remote=True,
                    supports_generation=True,
                    popularity_score=float(c["popularity"]),
                )
            )
        return selected

    def _supports_text_generation(self, model_row: dict, model_id: str) -> bool:
        architecture = model_row.get("architecture")
        output_modalities: list[str] = []
        if isinstance(architecture, dict):
            raw_modalities = architecture.get("output_modalities", [])
            if isinstance(raw_modalities, list):
                output_modalities = [str(v).lower() for v in raw_modalities]

        if output_modalities and "text" not in output_modalities:
            return False

        lowered = model_id.lower()
        if "embedding" in lowered or "/embed" in lowered:
            return False
        return True

    def _price_per_1k_chars(self, model_row: dict) -> float:
        pricing = model_row.get("pricing", {})
        if not isinstance(pricing, dict):
            return 0.002

        prompt_per_token = self._safe_float(pricing.get("prompt"), 0.0)
        completion_per_token = self._safe_float(pricing.get("completion"), 0.0)
        request_fixed = self._safe_float(pricing.get("request"), 0.0)

        # Rough conversion: 1 token ~= 4 chars.
        token_price_for_1k_chars = (prompt_per_token + completion_per_token) * 250.0
        total = token_price_for_1k_chars + request_fixed
        return max(total, 0.000001)

    def _extract_popularity(self, model_row: dict, index: int, total: int) -> float:
        for key in ("rank", "weekly_rank", "monthly_rank"):
            if key in model_row:
                rank = self._safe_float(model_row.get(key), -1.0)
                if rank > 0:
                    # rank=1 should be close to 1.0, then decays smoothly.
                    return max(0.0, min(1.0, 1.0 / (1.0 + (rank - 1.0) * 0.15)))

        for key in ("score", "rating"):
            if key in model_row:
                val = self._safe_float(model_row.get(key), -1.0)
                if val >= 0:
                    return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))

        for key in ("popularity", "usage", "usage_count", "request_count", "downloads"):
            if key in model_row:
                val = self._safe_float(model_row.get(key), -1.0)
                if val >= 0:
                    # Saturating transform to keep stable range 0..1
                    return min(1.0, val / (val + 1000.0))

        if total <= 1:
            return 0.5
        # Fallback: API order is treated as a popularity ranking proxy.
        return max(0.0, 1.0 - (index / (total - 1)))

    def _infer_complexity(self, model_row: dict, model_id: str) -> ModelComplexity:
        context_length = self._safe_float(model_row.get("context_length"), 0.0)
        lowered = model_id.lower()
        if "mini" in lowered or "nano" in lowered or "8b" in lowered:
            return ModelComplexity.SMALL
        if "70b" in lowered or "sonnet" in lowered or "opus" in lowered or context_length >= 200_000:
            return ModelComplexity.LARGE
        if context_length >= 100_000 or "pro" in lowered or "32b" in lowered:
            return ModelComplexity.MEDIUM
        return ModelComplexity.MEDIUM

    def _infer_specialty(self, model_row: dict, model_id: str) -> ModelSpecialty:
        lowered = model_id.lower()
        if "coder" in lowered or "code" in lowered:
            return ModelSpecialty.CODE
        if "reason" in lowered or "o1" in lowered or "o3" in lowered or "deepseek-r1" in lowered:
            return ModelSpecialty.REASONING
        return ModelSpecialty.CHAT

    def _openrouter_enabled(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    def _replace_local_models_from_llmfit_if_enabled(self) -> None:
        llmfit_url = os.getenv("LLMFIT_SERVICE_URL", "").strip()
        if not llmfit_url:
            return

        replace_enabled = os.getenv("LLMFIT_REPLACE_LOCAL_MODELS", "true").lower() in ("1", "true", "yes")
        if not replace_enabled:
            return

        dynamic_local_models = self._get_llmfit_local_models()
        if not dynamic_local_models:
            return

        remote_models = [m for m in self.models if m.is_remote]
        self.models = dynamic_local_models + remote_models
        print(f"[INFO] Replaced local catalog from LLMFit recommendations: {len(dynamic_local_models)} models")

    def _persist_llmfit_snapshot(self, models: list[ModelDefinition]) -> None:
        try:
            parent_dir = os.path.dirname(self._llmfit_snapshot_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            payload = []
            for m in models:
                payload.append(
                    {
                        "huggingface_id": m.huggingface_id,
                        "complexity": m.complexity.value,
                        "specialty": m.specialty.value,
                        "estimated_vram_gb": m.estimated_vram_gb,
                        "cost_per_1k_chars": m.cost_per_1k_chars,
                        "provider": m.provider.value,
                        "is_remote": m.is_remote,
                        "supports_generation": m.supports_generation,
                        "popularity_score": m.popularity_score,
                    }
                )
            with open(self._llmfit_snapshot_path, "w", encoding="utf-8") as f:
                json.dump({"models": payload, "updated_at": int(time.time())}, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to persist LLMFit snapshot: {e}")

    def _load_llmfit_snapshot(self) -> list[ModelDefinition]:
        try:
            if not os.path.exists(self._llmfit_snapshot_path):
                return []
            with open(self._llmfit_snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload.get("models", []) if isinstance(payload, dict) else []
            restored: list[ModelDefinition] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("huggingface_id", "")).strip()
                if not mid:
                    continue
                complexity_raw = str(row.get("complexity", "medium")).strip().lower()
                specialty_raw = str(row.get("specialty", "chat")).strip().lower()
                provider_raw = str(row.get("provider", "local")).strip().lower()
                complexity = {
                    "small": ModelComplexity.SMALL,
                    "medium": ModelComplexity.MEDIUM,
                    "large": ModelComplexity.LARGE,
                }.get(complexity_raw, ModelComplexity.MEDIUM)
                specialty = {
                    "general": ModelSpecialty.GENERAL,
                    "code": ModelSpecialty.CODE,
                    "chat": ModelSpecialty.CHAT,
                    "reasoning": ModelSpecialty.REASONING,
                }.get(specialty_raw, ModelSpecialty.CHAT)
                provider = ModelProvider.OPENROUTER if provider_raw == "openrouter" else ModelProvider.LOCAL
                restored.append(
                    ModelDefinition(
                        huggingface_id=mid,
                        complexity=complexity,
                        specialty=specialty,
                        estimated_vram_gb=self._safe_float(row.get("estimated_vram_gb"), 6.0),
                        cost_per_1k_chars=self._safe_float(row.get("cost_per_1k_chars"), 0.0001),
                        provider=provider,
                        is_remote=bool(row.get("is_remote", False)),
                        supports_generation=bool(row.get("supports_generation", True)),
                        popularity_score=self._safe_float(row.get("popularity_score"), 0.5),
                    )
                )
            return restored
        except Exception as e:
            print(f"[WARNING] Failed to load LLMFit snapshot fallback: {e}")
            return []

    def _get_llmfit_local_models(self) -> list[ModelDefinition]:
        now = time.time()
        if now < ModelCatalog._llmfit_local_cache_expires_at and ModelCatalog._llmfit_local_cache:
            return ModelCatalog._llmfit_local_cache

        models = self._fetch_llmfit_local_models()
        if models:
            ttl = int(os.getenv("LLMFIT_CACHE_TTL_SEC", "900"))
            ttl = max(60, min(ttl, 3600))
            ModelCatalog._llmfit_local_cache = models
            ModelCatalog._llmfit_local_cache_expires_at = now + ttl
            ModelCatalog._last_llmfit_local_fallback = list(models)
            self._persist_llmfit_snapshot(models)
            return models

        # If refresh failed, keep serving last known-good llmfit snapshot as fallback.
        if ModelCatalog._last_llmfit_local_fallback:
            print("[WARNING] Using last known-good LLMFit local catalog snapshot as fallback")
            return list(ModelCatalog._last_llmfit_local_fallback)
        disk_snapshot = self._load_llmfit_snapshot()
        if disk_snapshot:
            print("[WARNING] Using persisted LLMFit local catalog snapshot from disk as fallback")
            ModelCatalog._last_llmfit_local_fallback = list(disk_snapshot)
            return disk_snapshot
        return ModelCatalog._llmfit_local_cache

    def _fetch_llmfit_local_models(self) -> list[ModelDefinition]:
        llmfit_url = os.getenv("LLMFIT_SERVICE_URL", "").strip()
        if not llmfit_url:
            return []

        min_fit = os.getenv("LLMFIT_MIN_FIT", "good").strip().lower() or "good"
        limit = os.getenv("LLMFIT_LIMIT", "12").strip()
        try:
            limit_value = max(3, min(int(limit), 50))
        except ValueError:
            limit_value = 12

        query = urllib_parse.urlencode({
            "limit": limit_value,
            "min_fit": min_fit,
        })
        endpoint = f"{llmfit_url.rstrip('/')}/api/v1/models/top?{query}"

        try:
            with urllib_request.urlopen(endpoint, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as err:
            print(f"[WARNING] LLMFit replacement fetch failed at {endpoint}: {err}")
            return []

        rows = payload.get("models", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list) or not rows:
            return []

        local_models: list[ModelDefinition] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("name", "")).strip()
            if not model_id:
                continue
            if not self._is_llmfit_model_loadable(model_id):
                continue
            # Skip clearly non-chat generation models.
            lowered = model_id.lower()
            if "ocr" in lowered or "whisper" in lowered or "embedding" in lowered:
                continue

            category = str(row.get("category", "General")).strip().lower()
            fit_label = str(row.get("fit_level", "good")).strip().lower()
            memory_required = self._safe_float(row.get("memory_required_gb"), 0.0)
            score = self._safe_float(row.get("score"), 50.0)

            specialty = ModelSpecialty.CHAT
            if "coding" in category or "code" in category:
                specialty = ModelSpecialty.CODE
            elif "reason" in category:
                specialty = ModelSpecialty.REASONING
            elif "general" in category:
                specialty = ModelSpecialty.GENERAL

            complexity = ModelComplexity.MEDIUM
            if memory_required > 10.0:
                complexity = ModelComplexity.LARGE
            elif memory_required <= 5.0:
                complexity = ModelComplexity.SMALL

            # Lower pseudo-cost for better llmfit score + better fit.
            # Cost here is a routing proxy for local models (not billing).
            fit_bonus = 0.00002 if fit_label == "perfect" else 0.00005
            score_penalty = max(0.0, (100.0 - min(score, 100.0))) / 1_000_000
            proxy_cost = fit_bonus + score_penalty

            local_models.append(
                ModelDefinition(
                    huggingface_id=model_id,
                    complexity=complexity,
                    specialty=specialty,
                    estimated_vram_gb=max(0.1, memory_required) if memory_required > 0 else 6.0,
                    cost_per_1k_chars=proxy_cost,
                    provider=ModelProvider.LOCAL,
                    is_remote=False,
                    supports_generation=True,
                    popularity_score=max(0.05, min(1.0, score / 100.0)),
                )
            )

        if local_models:
            # Persist as static fallback snapshot for future outages.
            ModelCatalog._last_llmfit_local_fallback = list(local_models)
            self._persist_llmfit_snapshot(local_models)
        return local_models

    def _apply_model_overrides_from_env(self) -> None:
        """Allow explicit defaults without code changes."""
        # Format: comma-separated model IDs already present in catalog.
        preferred_order = os.getenv("YAGUARETE_PREFERRED_LOCAL_MODELS", "").strip()
        if not preferred_order:
            return

        preferred_ids = [mid.strip() for mid in preferred_order.split(",") if mid.strip()]
        if not preferred_ids:
            return

        priority: dict[str, int] = {mid: idx for idx, mid in enumerate(preferred_ids)}
        self.models.sort(key=lambda m: (priority.get(m.huggingface_id, 10_000), m.is_remote))

    def _prioritize_by_llmfit_if_enabled(self) -> None:
        """
        Optional integration with llmfit service.

        Expected env vars:
        - LLMFIT_SERVICE_URL=http://127.0.0.1:8787
        - LLMFIT_MIN_FIT=good|perfect|marginal (default: good)
        - LLMFIT_LIMIT=12 (default: 12)
        """
        llmfit_url = os.getenv("LLMFIT_SERVICE_URL", "").strip()
        if not llmfit_url:
            return

        min_fit = os.getenv("LLMFIT_MIN_FIT", "good").strip().lower() or "good"
        limit = os.getenv("LLMFIT_LIMIT", "12").strip()
        try:
            limit_value = max(1, min(int(limit), 50))
        except ValueError:
            limit_value = 12

        query = urllib_parse.urlencode({
            "limit": limit_value,
            "min_fit": min_fit,
        })
        endpoint = f"{llmfit_url.rstrip('/')}/api/v1/models/top?{query}"

        try:
            with urllib_request.urlopen(endpoint, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as err:
            print(f"[WARNING] LLMFit service unavailable at {endpoint}: {err}")
            return

        llmfit_models = payload.get("models", []) if isinstance(payload, dict) else []
        llmfit_ranking: list[str] = []
        for row in llmfit_models:
            if isinstance(row, dict):
                name = str(row.get("name", "")).strip()
                if name:
                    llmfit_ranking.append(name)

        if not llmfit_ranking:
            return

        rank = {model_id: idx for idx, model_id in enumerate(llmfit_ranking)}

        # Reorder only local models known by Yaguarete. Remote models keep their relative order.
        self.models.sort(
            key=lambda m: (
                m.is_remote,
                rank.get(m.huggingface_id, 10_000),
                m.complexity.value,
            )
        )
        print(
            "[INFO] Applied LLMFit ranking to local catalog. "
            f"Matched={len([m for m in self.models if m.huggingface_id in rank])}"
        )

    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        for model in self.models:
            if model.huggingface_id == model_id:
                return model
        return None

    def _generation_candidates(self, local_only: bool = True) -> list[ModelDefinition]:
        candidates = [m for m in self.models if not m.is_remote] if local_only else list(self.models)
        if not self._openrouter_enabled():
            candidates = [m for m in candidates if not m.is_remote]
        return [m for m in candidates if m.supports_generation]

    def find_best_model(self, required_complexity: ModelComplexity, required_specialty: ModelSpecialty, local_only: bool = True) -> Optional[ModelDefinition]:
        """
        Finds the best matching model based on complexity and specialty.

        Args:
            required_complexity: Required model complexity
            required_specialty: Required model specialty
            local_only: If True, only return local models; if False, include remote models
        """
        candidates = self._generation_candidates(local_only=local_only)

        # 1. Exact match
        for model in candidates:
            if model.complexity == required_complexity and model.specialty == required_specialty:
                return model

        # 2. Match complexity, fallback to general/chat
        for model in candidates:
            if model.complexity == required_complexity and model.specialty in [ModelSpecialty.GENERAL, ModelSpecialty.CHAT]:
                return model

        # 3. Match specialty with flexible complexity fallback
        for model in candidates:
            if model.specialty == required_specialty:
                return model

        # 4. Ultimate fallback: first available chat/general model
        for model in candidates:
            if model.specialty in [ModelSpecialty.GENERAL, ModelSpecialty.CHAT]:
                return model

        return candidates[0] if candidates else None

    def find_best_local_model(self, required_complexity: ModelComplexity, required_specialty: ModelSpecialty) -> Optional[ModelDefinition]:
        """Finds the best local model."""
        return self.find_best_model(required_complexity, required_specialty, local_only=True)

    def find_best_remote_model(self, required_complexity: ModelComplexity, required_specialty: ModelSpecialty) -> Optional[ModelDefinition]:
        """Finds the best remote (OpenRouter) model."""
        if not self._openrouter_enabled():
            return None
        candidates = [m for m in self.models if m.is_remote and m.supports_generation]

        # 1. Exact match
        for model in candidates:
            if model.complexity == required_complexity and model.specialty == required_specialty:
                return model

        # 2. Match complexity
        for model in candidates:
            if model.complexity == required_complexity:
                return model

        # 3. Fallback: cheapest model
        if candidates:
            return min(candidates, key=lambda m: m.cost_per_1k_chars)

        return None

    def get_default_model(self) -> ModelDefinition:
        """Returns a safe default model."""
        model = self.find_best_local_model(ModelComplexity.SMALL, ModelSpecialty.CHAT)
        if model is not None:
            return model
        for candidate in self._generation_candidates(local_only=False):
            return candidate
        raise RuntimeError("No generation-capable models configured")

    def get_all_local_models(self) -> list[ModelDefinition]:
        """Returns all local models."""
        return [m for m in self.models if not m.is_remote]

    def get_all_remote_models(self) -> list[ModelDefinition]:
        """Returns all remote models."""
        return [m for m in self.models if m.is_remote]

    def get_generation_models(self, local_only: bool = False) -> list[ModelDefinition]:
        return self._generation_candidates(local_only=local_only)
