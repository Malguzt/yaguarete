from typing import Callable, Dict, Optional, Tuple, Any
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .model_catalog import ModelComplexity

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError as e:
    print(f"[WARNING] BitsAndBytesConfig not found in transformers: {e}")
    HAS_BNB = False
    BitsAndBytesConfig = None
    # type: ignore


class ModelRuntimeLoader:
    """Single responsibility: choose device and load model/tokenizer with OOM fallback strategies."""

    def __init__(self, profiler) -> None:
        self.profiler = profiler
        self.enable_cpu_overflow = os.getenv("ENABLE_CPU_OVERFLOW", "1") == "1"
        self.force_auto_device_map = os.getenv("FORCE_AUTO_DEVICE_MAP", "0") == "1"
        self.trust_remote_code_default = os.getenv("TRUST_REMOTE_CODE", "0") == "1"
        self.trust_remote_code_allowlist = {
            m.strip() for m in os.getenv("TRUST_REMOTE_CODE_MODELS", "").split(",") if m.strip()
        }
        self._requires_remote_code_cache: dict[str, bool] = {}
        self.four_bit_memory_ratio = max(
            0.2, min(0.8, float(os.getenv("FOUR_BIT_MEMORY_RATIO", "0.38")))
        )

    @property
    def has_bnb(self) -> bool:
        return HAS_BNB

    def parse_cuda_index(self, device_label: str) -> Optional[int]:
        if not device_label.startswith("cuda"):
            return None
        parts = device_label.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    def infer_model_device(self, model: AutoModelForCausalLM) -> str:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

    def is_oom_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return isinstance(error, torch.OutOfMemoryError) or "out of memory" in text or "cuda out of memory" in text

    def _is_quantized_offload_validation_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return ("some modules are dispatched on the cpu or the disk" in text or 
                "_is_hf_initialized" in text or
                "params4bit" in text)

    def is_trust_remote_code_error(self, error: Exception) -> bool:
        text = str(error).lower()
        mentions_trust_flag = "trust_remote_code=true" in text or "trust_remote_code = true" in text
        requires_custom_code = (
            "contains custom code" in text
            or "requires you to execute the configuration file" in text
        )
        return mentions_trust_flag or requires_custom_code

    def _is_model_remote_code_allowed(self, huggingface_id: str) -> bool:
        return self.trust_remote_code_default or (huggingface_id in self.trust_remote_code_allowlist)

    def is_model_remote_code_allowed(self, huggingface_id: str) -> bool:
        return self._is_model_remote_code_allowed(huggingface_id)

    def model_requires_remote_code(self, huggingface_id: str) -> bool:
        cached = self._requires_remote_code_cache.get(huggingface_id)
        if cached is not None:
            return cached
        try:
            AutoConfig.from_pretrained(huggingface_id, trust_remote_code=False)
            self._requires_remote_code_cache[huggingface_id] = False
            return False
        except Exception as e:
            needs_remote_code = self.is_trust_remote_code_error(e)
            self._requires_remote_code_cache[huggingface_id] = needs_remote_code
            return needs_remote_code

    def choose_target_device(self, estimated_needed_gb: float, loaded_model_devices: dict[str, str]) -> str:
        if torch.cuda.is_available():
            gpu_info = self.profiler.get_gpu_vram_info()
            if not gpu_info:
                return "cuda:0"

            loaded_per_gpu: dict[int, int] = {gpu_id: 0 for gpu_id in gpu_info.keys()}
            for _, dev in loaded_model_devices.items():
                idx = self.parse_cuda_index(dev)
                if idx is not None and idx in loaded_per_gpu:
                    loaded_per_gpu[idx] += 1

            candidates = []
            for gpu_id, info in gpu_info.items():
                safe_gb = info.get("safe_limit_gb", 0.0)
                free_gb = info.get("free_gb", 0.0)
                # Prefer GPUs that can fit the model under safe limit, then lower load.
                fits_safely = 0 if safe_gb >= estimated_needed_gb else 1
                candidates.append((fits_safely, loaded_per_gpu[gpu_id], -safe_gb, -free_gb, gpu_id))

            candidates.sort()
            chosen_gpu_id = candidates[0][4]
            return f"cuda:{chosen_gpu_id}"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_load_kwargs(
        self,
        complexity: ModelComplexity,
        estimated_needed: float,
        target_device: str,
    ) -> tuple[dict, bool]:
        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            # Always explicit to avoid interactive trust resolution (breaks in background threads).
            "trust_remote_code": self.trust_remote_code_default,
        }

        use_4bit = False
        target_gpu_idx = self.parse_cuda_index(target_device) if target_device.startswith("cuda") else None
        if HAS_BNB and target_gpu_idx is not None:
            gpu_info = self.profiler.get_gpu_vram_info().get(target_gpu_idx, {})
            safe_gb = gpu_info.get("safe_limit_gb", 0.0)
            if estimated_needed > safe_gb and safe_gb > 0:
                use_4bit = True

        if use_4bit:
            print("[DEBUG] Loading with 4-bit quantization (bitsandbytes) due to VRAM budget...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # Required when quantized modules can be offloaded to CPU/disk.
                llm_int8_enable_fp32_cpu_offload=self.enable_cpu_overflow,
            )

        used_auto_device_map = False
        disable_multi_gpu_split = os.getenv("DISABLE_MULTI_GPU_SPLIT", "0") == "1"
        should_use_auto_map = (
            torch.cuda.is_available()
            and not disable_multi_gpu_split
            and (complexity in (ModelComplexity.MEDIUM, ModelComplexity.LARGE) or self.force_auto_device_map)
            and self.enable_cpu_overflow
        )
        if should_use_auto_map:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = self.profiler.generate_max_memory_mapping()
            load_kwargs["offload_folder"] = "/tmp/donmingo-offload"
            load_kwargs["offload_state_dict"] = True
            used_auto_device_map = True

        return load_kwargs, used_auto_device_map

    def estimate_effective_vram_need_gb(self, estimated_needed_gb: float, target_device: str) -> float:
        """
        Returns a more realistic VRAM estimate for admission checks.
        Catalog estimates are usually FP16-size; when 4-bit is expected, effective
        footprint tends to be significantly lower.
        """
        effective_needed = estimated_needed_gb
        if not target_device.startswith("cuda") or not HAS_BNB:
            return effective_needed

        target_gpu_idx = self.parse_cuda_index(target_device)
        if target_gpu_idx is None:
            return effective_needed

        gpu_info = self.profiler.get_gpu_vram_info().get(target_gpu_idx, {})
        safe_gb = gpu_info.get("safe_limit_gb", 0.0)
        if safe_gb > 0 and estimated_needed_gb > safe_gb:
            effective_needed = max(0.5, estimated_needed_gb * self.four_bit_memory_ratio)
        return effective_needed

    def _place_model(self, model: AutoModelForCausalLM, target_device: str, used_auto_device_map: bool) -> str:
        if used_auto_device_map or getattr(model, "hf_device_map", None):
            return "auto"
        if target_device.startswith("cuda"):
            gpu_idx = self.parse_cuda_index(target_device) or 0
            print(f"[DEBUG] Moving model to CUDA device {gpu_idx}...")
            torch.cuda.set_device(gpu_idx)
            model.to(torch.device(f"cuda:{gpu_idx}"))
            return f"cuda:{gpu_idx}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[DEBUG] Moving model to MPS...")
            model.to("mps")
            return "mps"
        print("[DEBUG] No accelerator available, keeping model on CPU")
        return "cpu"

    def _oom_emergency_retry(
        self,
        huggingface_id: str,
        prepare_for_retry: Callable[[], None],
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
        print(f"[WARNING] OOM loading {huggingface_id}. Retrying with emergency offload strategy...")
        prepare_for_retry()
        emergency_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": self.profiler.generate_max_memory_mapping(),
            "offload_folder": "/tmp/donmingo-offload",
            "offload_state_dict": True,
            "trust_remote_code": self.trust_remote_code_default,
        }
        if HAS_BNB:
            emergency_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(huggingface_id, **emergency_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(huggingface_id, trust_remote_code=self.trust_remote_code_default)
        model.eval()
        return model, tokenizer, "auto"

    def load_model_and_tokenizer(
        self,
        huggingface_id: str,
        complexity: ModelComplexity,
        estimated_needed: float,
        target_device: str,
        prepare_for_oom_retry: Callable[[], None],
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
        load_kwargs, used_auto_device_map = self._build_load_kwargs(
            complexity=complexity,
            estimated_needed=estimated_needed,
            target_device=target_device,
        )
        load_kwargs["trust_remote_code"] = self._is_model_remote_code_allowed(huggingface_id)
        print(f"[DEBUG] Calling AutoModelForCausalLM.from_pretrained() with kwargs: {load_kwargs}")
        try:
            model = AutoModelForCausalLM.from_pretrained(huggingface_id, **load_kwargs)
            print("[DEBUG] Model loading completed, moving to device...")
            model_device = self._place_model(model, target_device=target_device, used_auto_device_map=used_auto_device_map)
            tokenizer = AutoTokenizer.from_pretrained(huggingface_id, trust_remote_code=load_kwargs.get("trust_remote_code", False))
            model.eval()
            return model, tokenizer, model_device
        except Exception as e:
            text = str(e)
            needs_remote_code = self.is_trust_remote_code_error(e)
            can_trust_this_model = self._is_model_remote_code_allowed(huggingface_id)
            if needs_remote_code and can_trust_this_model and not load_kwargs.get("trust_remote_code", False):
                retry_kwargs = dict(load_kwargs)
                retry_kwargs["trust_remote_code"] = True
                print(
                    f"[WARNING] {huggingface_id} requires trust_remote_code. "
                    "Retrying because model is allowed by config..."
                )
                model = AutoModelForCausalLM.from_pretrained(huggingface_id, **retry_kwargs)
                model_device = self._place_model(
                    model,
                    target_device=target_device,
                    used_auto_device_map=used_auto_device_map,
                )
                tokenizer = AutoTokenizer.from_pretrained(huggingface_id, trust_remote_code=True)
                model.eval()
                return model, tokenizer, model_device
            # If 4-bit + offload validation fails, retry in 8-bit with explicit CPU offload enabled.
            if self._is_quantized_offload_validation_error(e) and HAS_BNB and self.enable_cpu_overflow:
                print(
                    "[WARNING] 4-bit quantized load rejected with CPU/disk dispatch. "
                    "Retrying with 8-bit + CPU offload..."
                )
                retry_kwargs = {
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    ),
                    "device_map": "auto",
                    "max_memory": self.profiler.generate_max_memory_mapping(),
                    "offload_folder": "/tmp/donmingo-offload",
                    "offload_state_dict": True,
                    "trust_remote_code": self.trust_remote_code_default,
                }
                model = AutoModelForCausalLM.from_pretrained(huggingface_id, **retry_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(
                    huggingface_id,
                    trust_remote_code=retry_kwargs.get("trust_remote_code", False),
                )
                model.eval()
                return model, tokenizer, "auto"
            if self.is_oom_error(e) and torch.cuda.is_available():
                return self._oom_emergency_retry(huggingface_id, prepare_for_retry=prepare_for_oom_retry)
            raise
