from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import threading
import torch
import time

# Optimize memory allocation to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import re
from typing import Optional, Dict, Any
from requests.exceptions import ConnectionError

from .hardware_profiler import HardwareProfiler
from .model_catalog import ModelCatalog, ModelComplexity, ModelProvider, ModelSpecialty
from .model_router import ModelRouter
from .model_size_estimator import ModelMemoryPredictor
from .model_artifact_manager import ModelArtifactManager
from .model_runtime_loader import ModelRuntimeLoader
from .openrouter_client import OpenRouterClient
from infrastructure.observability.metrics import (
    MODEL_ACTIVE_REQUESTS,
    MODEL_CACHE_SIZE,
    MODEL_GENERATION_SECONDS,
    MODEL_LOADED_INFO,
    MODEL_LOAD_TOTAL,
    MODEL_SELECTION_TOTAL,
    NODE_NAME,
)


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass


class ModelsHandler:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        if not cls._instance:
            cls._instance = super(ModelsHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Protects model/tokenizer caches and shared metadata.
        self._cache_lock = threading.RLock()
        # One lock per compute device to allow concurrency across GPUs.
        self._device_locks: dict[str, threading.Lock] = {}
        # One lock per model-id to avoid concurrent duplicate loads.
        self._model_load_locks: dict[str, threading.Lock] = {}

        memory_margin_percent = float(os.getenv("MODEL_MEMORY_MARGIN_PERCENT", "0.20"))
        self.profiler = HardwareProfiler(memory_margin_percent=memory_margin_percent)
        self.catalog = ModelCatalog()
        self.router = ModelRouter(self.catalog)
        self.predictor = ModelMemoryPredictor(catalog=self.catalog)

        self.artifact_manager = ModelArtifactManager(node_name=NODE_NAME)
        self.runtime_loader = ModelRuntimeLoader(profiler=self.profiler)

        # Cache of loaded models and tokenizers (key: huggingface_id)
        self._loaded_models: dict[str, AutoModelForCausalLM] = {}
        self._loaded_tokenizers: dict[str, AutoTokenizer] = {}
        self._loaded_model_devices: dict[str, str] = {}
        self._loaded_model_estimates_gb: dict[str, float] = {}
        self._model_last_used_at: dict[str, float] = {}
        self._blocked_models_until: dict[str, float] = {}
        self.model_failure_cooldown_sec = max(30.0, float(os.getenv("MODEL_FAILURE_COOLDOWN_SEC", "300")))

        # OpenRouter client (lazy initialization)
        self._openrouter_client: Optional[OpenRouterClient] = None
        self.enable_lru_eviction = os.getenv("ENABLE_LRU_EVICTION", "1") == "1"
        self.force_unload_all_for_large = os.getenv("FORCE_UNLOAD_ALL_FOR_LARGE", "0") == "1"
        self.vram_admission_buffer_gb = float(os.getenv("VRAM_ADMISSION_BUFFER_GB", "0.35"))
        self.auto_device_split_factor = max(0.1, min(1.0, float(os.getenv("AUTO_DEVICE_SPLIT_FACTOR", "0.5"))))

        # Preload flag
        self._preload_started = False
        self._preload_done = False

        # Print profile on startup
        print("--- Hardware Profile ---")
        print(self.profiler.get_profile_summary())
        print(f"BitsAndBytes (4-bit) support: {self.runtime_loader.has_bnb}")
        print("------------------------")

    def _free_memory(self) -> None:
        """Forces garbage collection and empties CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_openrouter_client(self) -> OpenRouterClient:
        """Get or create OpenRouter client (lazy initialization)."""
        if self._openrouter_client is None:
            try:
                self._openrouter_client = OpenRouterClient()
                print("[INFO] OpenRouter client initialized")
            except ValueError as e:
                print(f"[WARNING] OpenRouter not configured: {e}")
                raise
        return self._openrouter_client

    def _complexity_label_for_model(self, huggingface_id: str) -> str:
        for model_def in self.catalog.models:
            if model_def.huggingface_id == huggingface_id:
                return model_def.complexity.value
        return "unknown"

    def _get_device_lock(self, device_label: str) -> threading.Lock:
        with self._cache_lock:
            if device_label not in self._device_locks:
                self._device_locks[device_label] = threading.Lock()
            return self._device_locks[device_label]

    def _get_model_load_lock(self, huggingface_id: str) -> threading.Lock:
        with self._cache_lock:
            if huggingface_id not in self._model_load_locks:
                self._model_load_locks[huggingface_id] = threading.Lock()
            return self._model_load_locks[huggingface_id]

    def _prepare_for_oom_retry(self) -> None:
        self._unload_all_models()
        self._free_memory()

    def _device_safe_limit_gb(self, device_label: str) -> Optional[float]:
        if not device_label.startswith("cuda"):
            return None
        idx = self.runtime_loader.parse_cuda_index(device_label)
        if idx is None:
            return None
        gpu_info = self.profiler.get_gpu_vram_info().get(idx)
        if not gpu_info:
            return None
        return gpu_info.get("safe_limit_gb")

    def _touch_model(self, model_id: str) -> None:
        self._model_last_used_at[model_id] = time.time()

    def _is_model_temporarily_blocked(self, model_id: str) -> bool:
        unblock_at = self._blocked_models_until.get(model_id)
        if unblock_at is None:
            return False
        if time.time() >= unblock_at:
            self._blocked_models_until.pop(model_id, None)
            return False
        return True

    def _block_model_temporarily(self, model_id: str, reason: str) -> None:
        unblock_at = time.time() + self.model_failure_cooldown_sec
        self._blocked_models_until[model_id] = unblock_at
        print(
            f"[WARNING] Temporarily blocking model {model_id} for "
            f"{self.model_failure_cooldown_sec:.0f}s. reason={reason}"
        )

    def _pick_local_fallback(
        self,
        preferred_complexity: ModelComplexity,
        preferred_specialty: ModelSpecialty,
        excluded_model_ids: set[str],
    ) -> Optional[Any]:
        candidates = self.catalog.get_generation_models(local_only=True)
        if not candidates:
            return None

        def _eligible(model_def: Any) -> bool:
            return (
                model_def.huggingface_id not in excluded_model_ids
                and not self._is_model_temporarily_blocked(model_def.huggingface_id)
            )

        for model_def in candidates:
            if _eligible(model_def) and model_def.complexity == preferred_complexity and model_def.specialty == preferred_specialty:
                return model_def
        for model_def in candidates:
            if _eligible(model_def) and model_def.complexity == preferred_complexity:
                return model_def
        for model_def in candidates:
            if _eligible(model_def):
                return model_def
        return None

    def _estimate_reserved_for_target_gb(self, target_device: str) -> float:
        if not self._loaded_model_estimates_gb:
            return 0.0

        reserved = 0.0
        gpu_count = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        for model_id, estimate in self._loaded_model_estimates_gb.items():
            loaded_device = self._loaded_model_devices.get(model_id, "unknown")
            if target_device.startswith("cuda"):
                if loaded_device == target_device:
                    reserved += estimate
                elif loaded_device == "auto":
                    reserved += estimate * self.auto_device_split_factor
            elif target_device == "auto":
                if loaded_device.startswith("cuda") or loaded_device == "auto":
                    portion = estimate if loaded_device != "auto" else (estimate / gpu_count)
                    reserved += portion
            else:
                # CPU target does not need GPU VRAM budget.
                continue
        return reserved

    def _unload_model(self, model_id: str) -> None:
        model_device = self._loaded_model_devices.get(model_id, "unknown")
        MODEL_LOADED_INFO.labels(
            model_id=model_id,
            complexity=self._complexity_label_for_model(model_id),
            device=model_device,
            node=NODE_NAME,
        ).set(0)
        self._loaded_models.pop(model_id, None)
        self._loaded_tokenizers.pop(model_id, None)
        self._loaded_model_devices.pop(model_id, None)
        self._loaded_model_estimates_gb.pop(model_id, None)
        self._model_last_used_at.pop(model_id, None)
        MODEL_CACHE_SIZE.labels(node=NODE_NAME).set(len(self._loaded_models))

    def _evict_models_for_budget(self, required_gb: float, target_device: str, preserve_model_id: str) -> None:
        if not self.enable_lru_eviction or required_gb <= 0:
            return

        if target_device.startswith("cuda"):
            budget_limit = self._device_safe_limit_gb(target_device)
        else:
            budget_limit = self.profiler.get_total_available_vram_gb()

        if budget_limit is None or budget_limit <= 0:
            return

        required_with_buffer = required_gb + max(0.0, self.vram_admission_buffer_gb)
        reserved = self._estimate_reserved_for_target_gb(target_device)
        if reserved + required_with_buffer <= budget_limit:
            return

        with self._cache_lock:
            candidates = []
            for model_id in self._loaded_models.keys():
                if model_id == preserve_model_id:
                    continue
                last_used = self._model_last_used_at.get(model_id, 0.0)
                est = self._loaded_model_estimates_gb.get(model_id, 0.0)
                candidates.append((last_used, -est, model_id))

            if not candidates:
                return

            candidates.sort()
            evicted = []
            for _, _, model_id in candidates:
                current_reserved = self._estimate_reserved_for_target_gb(target_device)
                if current_reserved + required_with_buffer <= budget_limit:
                    break
                print(
                    f"[EVICT] Unloading LRU model {model_id} to free VRAM for "
                    f"{preserve_model_id} on {target_device}"
                )
                self._unload_model(model_id)
                evicted.append(model_id)

            if evicted:
                self._free_memory()

    def preload_models(self) -> None:
        """Preload default models at startup in background."""
        if self._preload_started:
            return

        self._preload_started = True
        print("[INFO] Starting background model preload...")

        def _preload_worker() -> None:
            try:
                preload_candidates = []
                available_vram = self.profiler.get_total_available_vram_gb()

                # Only preload medium models when there is enough safe VRAM.
                if torch.cuda.is_available() and available_vram >= 12.0:
                    preload_candidates.extend([
                        ("Qwen/Qwen2.5-7B-Instruct", ModelComplexity.MEDIUM),
                        ("Qwen/Qwen2.5-Coder-7B", ModelComplexity.MEDIUM),
                    ])

                small_default = self.catalog.find_best_local_model(
                    ModelComplexity.SMALL,
                    ModelSpecialty.CHAT,
                )
                if small_default is not None:
                    preload_candidates.append((small_default.huggingface_id, small_default.complexity))

                deduped_candidates = []
                seen = set()
                for model_id, complexity in preload_candidates:
                    if model_id in seen:
                        continue
                    seen.add(model_id)
                    deduped_candidates.append((model_id, complexity))
                preload_candidates = deduped_candidates

                success_count = 0
                for model_id, complexity in preload_candidates:
                    try:
                        print(f"[DEBUG] Preloading model: {model_id} ({complexity.value})")
                        self.get_model_and_tokenizer(model_id, complexity)
                        success_count += 1
                    except Exception as model_err:
                        print(f"[WARNING] Preload model failed ({model_id}): {model_err}")

                if success_count == 0 and small_default is not None and torch.cuda.is_available():
                    try:
                        print("[WARNING] Falling back to SMALL preload due to previous failures...")
                        self.get_model_and_tokenizer(small_default.huggingface_id, small_default.complexity)
                        success_count += 1
                    except Exception as fallback_err:
                        print(f"[ERROR] Fallback preload failed: {fallback_err}")

                print(f"[DEBUG] Model preload sequence completed. successful={success_count}")
                self._preload_done = True
            except Exception as e:
                print(f"[ERROR] Preload failed: {e}")
                self._preload_done = True

        preload_thread = threading.Thread(target=_preload_worker, daemon=True, name="ModelPreloader")
        preload_thread.start()

    def _unload_all_models(self) -> None:
        """Unload all currently loaded models to free memory."""
        print("[DEBUG] Unloading all models to free memory...")
        for model_id, device in self._loaded_model_devices.items():
            MODEL_LOADED_INFO.labels(
                model_id=model_id,
                complexity=self._complexity_label_for_model(model_id),
                device=device,
                node=NODE_NAME,
            ).set(0)
        self._loaded_models.clear()
        self._loaded_tokenizers.clear()
        self._loaded_model_devices.clear()
        self._loaded_model_estimates_gb.clear()
        self._model_last_used_at.clear()
        MODEL_CACHE_SIZE.labels(node=NODE_NAME).set(0)
        self._free_memory()

    def get_model_and_tokenizer(self, huggingface_id: str, complexity: ModelComplexity) -> Any:
        """Load and return model/tokenizer, managing memory dynamically."""
        complexity_label = complexity.value if isinstance(complexity, ModelComplexity) else "unknown"
        model_load_lock = self._get_model_load_lock(huggingface_id)
        lock_acquired = model_load_lock.acquire(timeout=600)
        if not lock_acquired:
            raise TimeoutException(f"Timed out waiting to load model lock: {huggingface_id}")
        try:
            with self._cache_lock:
                if huggingface_id in self._loaded_models:
                    print(f"[DEBUG] Model {huggingface_id} already loaded in cache")
                    self._touch_model(huggingface_id)
                    cached_device = self._loaded_model_devices.get(
                        huggingface_id,
                        self.runtime_loader.infer_model_device(self._loaded_models[huggingface_id]),
                    )
                    MODEL_LOADED_INFO.labels(
                        model_id=huggingface_id,
                        complexity=complexity_label,
                        device=cached_device,
                        node=NODE_NAME,
                    ).set(1)
                    MODEL_CACHE_SIZE.labels(node=NODE_NAME).set(len(self._loaded_models))
                    MODEL_LOAD_TOTAL.labels(
                        model_id=huggingface_id,
                        complexity=complexity_label,
                        device=cached_device,
                        node=NODE_NAME,
                        status="cache_hit",
                    ).inc()
                    return self._loaded_models[huggingface_id], self._loaded_tokenizers[huggingface_id]

                if complexity == ModelComplexity.LARGE and self.force_unload_all_for_large:
                    self._unload_all_models()

            if (
                self.runtime_loader.model_requires_remote_code(huggingface_id)
                and not self.runtime_loader.is_model_remote_code_allowed(huggingface_id)
            ):
                raise ValueError(
                    f"Loading {huggingface_id} requires trust_remote_code=True but it is not allowed by current config"
                )

            available_vram = self.profiler.get_total_available_vram_gb()
            estimated_needed = self.predictor.estimate_vram_required_gb(huggingface_id, target_dtype="float16")
            target_device = self.runtime_loader.choose_target_device(
                estimated_needed_gb=estimated_needed,
                loaded_model_devices=self._loaded_model_devices,
            )
            effective_needed = self.runtime_loader.estimate_effective_vram_need_gb(
                estimated_needed_gb=estimated_needed,
                target_device=target_device,
            )
            target_device_safe = self._device_safe_limit_gb(target_device)

            print(
                f"[INFO] Memory Check: {huggingface_id} needs ~{estimated_needed:.2f}GB "
                f"(effective ~{effective_needed:.2f}GB). "
                f"Available VRAM safe total: {available_vram:.2f}GB. Target device: {target_device} "
                f"(safe: {target_device_safe:.2f}GB)."
                if target_device_safe is not None
                else f"[INFO] Memory Check: {huggingface_id} needs ~{estimated_needed:.2f}GB "
                f"(effective ~{effective_needed:.2f}GB). "
                f"Available VRAM safe total: {available_vram:.2f}GB. Target device: {target_device}."
            )
            admission_limit = target_device_safe if target_device_safe is not None else available_vram
            if effective_needed > admission_limit:
                print(
                    f"[WARNING] Model {huggingface_id} may exceed safe VRAM budget "
                    f"(effective {effective_needed:.2f}GB > limit {admission_limit:.2f}GB)."
                )

            self._evict_models_for_budget(
                required_gb=effective_needed,
                target_device=target_device,
                preserve_model_id=huggingface_id,
            )

            try:
                print(f"[DEBUG] Downloading model artifacts (if needed) for {huggingface_id}...")
                self.artifact_manager.ensure_local_artifacts(huggingface_id)
                print(f"[DEBUG] Model {huggingface_id} is ready locally.")
            except ConnectionError as e:
                print(f"[WARNING] Connection error while downloading {huggingface_id}: {e}")
                print("[INFO] Attempting local-cache fallback...")
                self.artifact_manager.try_local_fallback(huggingface_id)
            except Exception as e:
                print(f"[WARNING] Could not verify/download model {huggingface_id}. Error: {e}")
                print("[INFO] Attempting local-cache fallback...")
                self.artifact_manager.try_local_fallback(huggingface_id)

            model_device = target_device
            try:
                model, tokenizer, model_device = self.runtime_loader.load_model_and_tokenizer(
                    huggingface_id=huggingface_id,
                    complexity=complexity,
                    estimated_needed=estimated_needed,
                    target_device=target_device,
                    prepare_for_oom_retry=self._prepare_for_oom_retry,
                )
            except Exception as e:
                MODEL_LOAD_TOTAL.labels(
                    model_id=huggingface_id,
                    complexity=complexity_label,
                    device=model_device,
                    node=NODE_NAME,
                    status="error",
                ).inc()
                print(f"[ERROR] Failed to load model {huggingface_id}: {e}")
                import traceback
                traceback.print_exc()
                raise

            with self._cache_lock:
                self._loaded_models[huggingface_id] = model
                self._loaded_tokenizers[huggingface_id] = tokenizer
                self._loaded_model_devices[huggingface_id] = model_device
                self._loaded_model_estimates_gb[huggingface_id] = estimated_needed
                self._touch_model(huggingface_id)

            MODEL_LOADED_INFO.labels(
                model_id=huggingface_id,
                complexity=complexity_label,
                device=model_device,
                node=NODE_NAME,
            ).set(1)
            MODEL_CACHE_SIZE.labels(node=NODE_NAME).set(len(self._loaded_models))
            MODEL_LOAD_TOTAL.labels(
                model_id=huggingface_id,
                complexity=complexity_label,
                device=model_device,
                node=NODE_NAME,
                status="success",
            ).inc()

            print("[DEBUG] Model and tokenizer cached")
            return model, tokenizer
        finally:
            model_load_lock.release()

    def generate_text(
        self,
        prompt: str,
        required_complexity: Optional[ModelComplexity] = None,
        model_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using the appropriate model (local or remote via OpenRouter).
        Uses a per-device lock to allow parallel generation across multiple GPUs.
        """
        print("[DEBUG] generate_text() called")
        selected_model_id = None
        selected_complexity = "unknown"
        selected_complexity_enum = None
        generation_started_at = None
        generation_status = "error"
        active_request_incremented = False
        device_lock = None
        lock_acquired = False
        try:
            if model_id:
                model_def = self.catalog.get_model(model_id)
                if model_def is None:
                    raise ValueError(f"Unknown model_id: {model_id}")
                if not model_def.supports_generation:
                    raise ValueError(f"Model {model_id} does not support text generation")
            else:
                model_def = self.router.route_prompt(prompt, required_complexity)
            selected_model_id = model_def.huggingface_id
            selected_complexity = model_def.complexity.value
            selected_complexity_enum = model_def.complexity
            generation_started_at = time.perf_counter()

            MODEL_SELECTION_TOTAL.labels(
                model_id=selected_model_id,
                complexity=selected_complexity,
                specialty=model_def.specialty.value,
            ).inc()

            # --- OPENROUTER HANDLING ---
            if model_def.is_remote:
                print(f"[INFO] Using remote model via OpenRouter: {selected_model_id}")
                try:
                    openrouter = self._get_openrouter_client()
                    response_text = openrouter.generate_text(
                        prompt=prompt,
                        model_id=selected_model_id,
                        max_tokens=max_new_tokens or 128,
                        temperature=temperature,
                    )
                    generation_status = "success"
                    return response_text
                except ValueError:
                    print(f"[WARNING] OpenRouter API key not configured, falling back to local models")
                    # Fall through to local model logic
                except Exception as e:
                    print(f"[ERROR] OpenRouter call failed: {e}, attempting fallback to local model")
                    # Fall through to local model logic
                fallback_model = self.catalog.find_best_local_model(model_def.complexity, model_def.specialty)
                if fallback_model is None:
                    fallback_model = self.catalog.get_default_model()
                print(f"[INFO] Falling back to local model: {fallback_model.huggingface_id}")
                model_def = fallback_model
                selected_model_id = model_def.huggingface_id
                selected_complexity = model_def.complexity.value
                selected_complexity_enum = model_def.complexity

            if self._is_model_temporarily_blocked(model_def.huggingface_id):
                fallback_model = self._pick_local_fallback(
                    preferred_complexity=model_def.complexity,
                    preferred_specialty=model_def.specialty,
                    excluded_model_ids={model_def.huggingface_id},
                )
                if fallback_model is not None:
                    print(
                        f"[WARNING] Selected model {model_def.huggingface_id} is temporarily blocked. "
                        f"Using fallback {fallback_model.huggingface_id}."
                    )
                    model_def = fallback_model
                    selected_model_id = fallback_model.huggingface_id
                    selected_complexity = fallback_model.complexity.value
                    selected_complexity_enum = fallback_model.complexity
                else:
                    raise RuntimeError(
                        f"Model {model_def.huggingface_id} is temporarily blocked and no fallback is available"
                    )

            # --- LOCAL MODEL HANDLING ---
            attempted_model_ids: set[str] = set()
            max_load_attempts = max(1, int(os.getenv("MODEL_LOAD_MAX_ATTEMPTS", "6")))
            model = None
            tokenizer = None
            current_model_def = model_def

            for _ in range(max_load_attempts):
                attempted_model_ids.add(current_model_def.huggingface_id)
                selected_model_id = current_model_def.huggingface_id
                selected_complexity = current_model_def.complexity.value
                selected_complexity_enum = current_model_def.complexity
                try:
                    model, tokenizer = self.get_model_and_tokenizer(
                        current_model_def.huggingface_id,
                        current_model_def.complexity,
                    )
                    break
                except Exception as load_error:
                    if self.runtime_loader.is_trust_remote_code_error(load_error):
                        self._block_model_temporarily(current_model_def.huggingface_id, reason="requires trust_remote_code")
                    elif self.runtime_loader.is_oom_error(load_error):
                        self._block_model_temporarily(current_model_def.huggingface_id, reason="oom")
                    else:
                        self._block_model_temporarily(
                            current_model_def.huggingface_id,
                            reason=f"load_error:{type(load_error).__name__}",
                        )

                    fallback_model = self._pick_local_fallback(
                        preferred_complexity=current_model_def.complexity,
                        preferred_specialty=current_model_def.specialty,
                        excluded_model_ids=attempted_model_ids,
                    )
                    if fallback_model is None:
                        raise

                    print(
                        f"[WARNING] Model load failed for {current_model_def.huggingface_id}. "
                        f"Retrying with fallback {fallback_model.huggingface_id}."
                    )
                    MODEL_SELECTION_TOTAL.labels(
                        model_id=fallback_model.huggingface_id,
                        complexity=fallback_model.complexity.value,
                        specialty=fallback_model.specialty.value,
                    ).inc()
                    current_model_def = fallback_model
            else:
                raise RuntimeError(
                    "No eligible local fallback model could be loaded after multiple attempts"
                )

            if model is None or tokenizer is None:
                raise RuntimeError("Model loading failed after fallback attempts")

            MODEL_ACTIVE_REQUESTS.labels(
                model_id=selected_model_id,
                complexity=selected_complexity,
            ).inc()
            active_request_incremented = True

            model_device_label = self.runtime_loader.infer_model_device(model)
            device_lock = self._get_device_lock(model_device_label)
            lock_acquired = device_lock.acquire(timeout=8)
            if not lock_acquired:
                generation_status = "timeout"
                print(f"[WARNING] Device {model_device_label} is busy.")
                return "I apologize, the selected model device is busy. Please try again."

            clean_prompt = re.sub(r'<[^>]+>', '', prompt).strip()
            if not clean_prompt:
                clean_prompt = "Hello"

            inputs = tokenizer(clean_prompt, return_tensors="pt", max_length=512, truncation=True)

            # Check if model uses device_map="auto" (multi-GPU distribution)
            has_device_map = getattr(model, "hf_device_map", None) is not None
            
            if has_device_map:
                # For models with device_map="auto", inputs should stay on CPU
                # model.generate() will handle moving them to the correct device
                print("[DEBUG] Model uses device_map='auto', keeping inputs on CPU for device distribution")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            else:
                # For single-device models, move inputs to the model's device
                try:
                    target_torch_device = next(model.parameters()).device
                except Exception:
                    target_torch_device = torch.device("cpu")
                inputs = {k: v.to(target_torch_device) for k, v in inputs.items()}

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            max_tokens = max_new_tokens if max_new_tokens is not None else (40 if selected_complexity_enum == ModelComplexity.LARGE else 32)
            max_tokens = max(8, min(int(max_tokens), 512))

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                )

            result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            self._free_memory()
            generation_status = "success"
            return result.strip()
        except TimeoutException as e:
            generation_status = "timeout"
            print(f"[ERROR] Operation timed out: {e}")
            raise RuntimeError("Generation timed out") from e
        except Exception as e:
            generation_status = "error"
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Generation failed") from e
        finally:
            if lock_acquired and device_lock is not None:
                device_lock.release()
            if selected_model_id and generation_started_at is not None:
                elapsed = time.perf_counter() - generation_started_at
                MODEL_GENERATION_SECONDS.labels(
                    model_id=selected_model_id,
                    complexity=selected_complexity,
                    status=generation_status,
                ).observe(elapsed)
                if active_request_incremented:
                    MODEL_ACTIVE_REQUESTS.labels(
                        model_id=selected_model_id,
                        complexity=selected_complexity,
                    ).dec()
