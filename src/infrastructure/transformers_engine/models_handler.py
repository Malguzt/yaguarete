from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import threading
import torch
import time

# Optimize memory allocation to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import re
from typing import Optional, Dict
from requests.exceptions import ConnectionError

from .hardware_profiler import HardwareProfiler
from .model_catalog import ModelCatalog, ModelComplexity
from .model_router import ModelRouter
from .model_size_estimator import ModelMemoryPredictor
from .model_artifact_manager import ModelArtifactManager
from .model_runtime_loader import ModelRuntimeLoader
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

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelsHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Protects model/tokenizer caches and shared metadata.
        self._cache_lock = threading.RLock()
        # One lock per compute device to allow concurrency across GPUs.
        self._device_locks: Dict[str, threading.Lock] = {}

        memory_margin_percent = float(os.getenv("MODEL_MEMORY_MARGIN_PERCENT", "0.20"))
        self.profiler = HardwareProfiler(memory_margin_percent=memory_margin_percent)
        self.catalog = ModelCatalog()
        self.router = ModelRouter(self.catalog)
        self.predictor = ModelMemoryPredictor(catalog=self.catalog)

        self.artifact_manager = ModelArtifactManager(node_name=NODE_NAME)
        self.runtime_loader = ModelRuntimeLoader(profiler=self.profiler)

        # Cache of loaded models and tokenizers (key: huggingface_id)
        self._loaded_models: Dict[str, AutoModelForCausalLM] = {}
        self._loaded_tokenizers: Dict[str, AutoTokenizer] = {}
        self._loaded_model_devices: Dict[str, str] = {}
        self._loaded_model_estimates_gb: Dict[str, float] = {}

        # Preload flag
        self._preload_started = False
        self._preload_done = False

        # Print profile on startup
        print("--- Hardware Profile ---")
        print(self.profiler.get_profile_summary())
        print(f"BitsAndBytes (4-bit) support: {self.runtime_loader.has_bnb}")
        print("------------------------")

    def _free_memory(self):
        """Forces garbage collection and empties CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _prepare_for_oom_retry(self) -> None:
        self._unload_all_models()
        self._free_memory()

    def preload_models(self) -> None:
        """Preload default models at startup in background."""
        if self._preload_started:
            return

        self._preload_started = True
        print("[INFO] Starting background model preload...")

        def _preload_worker():
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
                    preload_candidates = [
                        ("Qwen/Qwen2.5-7B-Instruct", ModelComplexity.MEDIUM),
                        ("Qwen/Qwen2.5-Coder-7B", ModelComplexity.MEDIUM),
                    ]
                else:
                    preload_candidates = [
                        ("Qwen/Qwen2.5-1.5B-Instruct", ModelComplexity.SMALL),
                    ]

                success_count = 0
                for model_id, complexity in preload_candidates:
                    try:
                        print(f"[DEBUG] Preloading model: {model_id} ({complexity.value})")
                        self.get_model_and_tokenizer(model_id, complexity)
                        success_count += 1
                    except Exception as model_err:
                        print(f"[WARNING] Preload model failed ({model_id}): {model_err}")

                if success_count == 0 and torch.cuda.is_available():
                    try:
                        print("[WARNING] Falling back to SMALL preload due to previous failures...")
                        self.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", ModelComplexity.SMALL)
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

    def _unload_all_models(self):
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
        MODEL_CACHE_SIZE.labels(node=NODE_NAME).set(0)
        self._free_memory()

    def get_model_and_tokenizer(self, huggingface_id: str, complexity: ModelComplexity):
        """Load and return model/tokenizer, managing memory dynamically."""
        complexity_label = complexity.value if isinstance(complexity, ModelComplexity) else "unknown"

        with self._cache_lock:
            if huggingface_id in self._loaded_models:
                print(f"[DEBUG] Model {huggingface_id} already loaded in cache")
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

            if complexity == ModelComplexity.LARGE:
                self._unload_all_models()

        available_vram = self.profiler.get_total_available_vram_gb()
        estimated_needed = self.predictor.estimate_vram_required_gb(huggingface_id, target_dtype="float16")
        target_device = self.runtime_loader.choose_target_device(
            estimated_needed_gb=estimated_needed,
            loaded_model_devices=self._loaded_model_devices,
        )

        print(
            f"[INFO] Memory Check: {huggingface_id} needs ~{estimated_needed:.2f}GB. "
            f"Available VRAM (safe total): {available_vram:.2f}GB. Target device: {target_device}"
        )
        if estimated_needed > available_vram:
            print(
                f"[WARNING] Model {huggingface_id} is predicted to exceed available VRAM "
                f"({estimated_needed:.2f}GB > {available_vram:.2f}GB)."
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

    def generate_text(self, prompt: str, required_complexity: Optional[ModelComplexity] = None) -> str:
        """
        Generate text using the appropriate model.
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

            try:
                model, tokenizer = self.get_model_and_tokenizer(model_def.huggingface_id, model_def.complexity)
            except Exception as load_error:
                if self.runtime_loader.is_oom_error(load_error) and model_def.complexity != ModelComplexity.SMALL:
                    fallback_model = self.catalog.find_best_model(ModelComplexity.SMALL, model_def.specialty)
                    if fallback_model is None:
                        fallback_model = self.catalog.get_default_model()
                    print(
                        f"[WARNING] OOM with {model_def.huggingface_id}. "
                        f"Falling back to {fallback_model.huggingface_id}."
                    )
                    selected_model_id = fallback_model.huggingface_id
                    selected_complexity = fallback_model.complexity.value
                    selected_complexity_enum = fallback_model.complexity
                    MODEL_SELECTION_TOTAL.labels(
                        model_id=selected_model_id,
                        complexity=selected_complexity,
                        specialty=fallback_model.specialty.value,
                    ).inc()
                    model, tokenizer = self.get_model_and_tokenizer(
                        fallback_model.huggingface_id,
                        fallback_model.complexity,
                    )
                else:
                    raise

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

            try:
                target_torch_device = next(model.parameters()).device
            except Exception:
                target_torch_device = torch.device("cpu")
            inputs = {k: v.to(target_torch_device) for k, v in inputs.items()}

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            max_tokens = 40 if selected_complexity_enum == ModelComplexity.LARGE else 32

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=pad_token_id,
                )

            result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            self._free_memory()
            generation_status = "success"
            return result.strip()
        except TimeoutException as e:
            generation_status = "timeout"
            print(f"[ERROR] Operation timed out: {e}")
            return "I apologize, the operation took too long. Please try again."
        except Exception as e:
            generation_status = "error"
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, I'm having trouble generating a response right now."
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
