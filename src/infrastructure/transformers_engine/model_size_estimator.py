from huggingface_hub import HfApi, hf_hub_download
import torch
import json
import os
from typing import Optional

class ModelMemoryPredictor:
    """Predicts required VRAM for models based on Hugging Face Hub metadata and local catalog."""

    def __init__(self, catalog=None):
        self.api = HfApi()
        self.catalog = catalog

    def estimate_vram_required_gb(self, model_id: str, target_dtype: str = "float16") -> float:
        """
        Estimates the VRAM required to load and run a model.
        """
        # 1. Try Catalog First (Manual overrides are most reliable)
        if self.catalog:
            for model_def in self.catalog.models:
                if model_def.huggingface_id == model_id:
                    print(f"[DEBUG] Using catalog estimate for {model_id}: {model_def.estimated_vram_gb} GB")
                    return model_def.estimated_vram_gb

        # 2. Try Hugging Face Metadata
        try:
            print(f"[DEBUG] Fetching metadata for {model_id} from Hugging Face Hub...")
            info = self.api.model_info(model_id)
            
            base_size_bytes = 0
            if hasattr(info, 'safetensors') and info.safetensors is not None:
                base_size_bytes = info.safetensors.get('total', 0)
            
            # If we have a size, use it but adjust for BF16 if it seems too small for a 7B model
            # (Sometimes 'total' is just one shard or compressed size)
            if base_size_bytes > 0:
                base_gb = base_size_bytes / (1024 ** 3)
                # If disk size is ~7GB for a 7B model, it's likely INT8 or BF16-compressed.
                # If we want FP16, we should double it if we suspect it's INT8.
                # But let's be simpler: if we have 'config.json', calculate from params.
                try:
                    config_path = hf_hub_download(model_id, "config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Heuristic for parameter count from config
                    hidden = config.get("hidden_size")
                    layers = config.get("num_hidden_layers")
                    vocab = config.get("vocab_size")
                    intermediate = config.get("intermediate_size", hidden * 4)
                    
                    if hidden and layers and vocab:
                        # Params ~= vocab*hidden + layers * (4*hidden^2 + 3*hidden*intermediate)
                        params = vocab * hidden + layers * (4 * (hidden**2) + 3 * hidden * intermediate)
                        print(f"[DEBUG] Calculated ~{params/1e9:.2f}B parameters from config.")
                        
                        # 2 bytes per param for FP16
                        runtime_gb = (params * 2) / (1024 ** 3)
                    else:
                        runtime_gb = base_gb
                except:
                    runtime_gb = base_gb
            else:
                runtime_gb = 16.0 # Fallback 7B
                
            # Adjust for target dtype
            multiplier = 1.0
            if target_dtype == "float32": multiplier = 2.0
            elif target_dtype == "int8": multiplier = 0.5
            elif target_dtype == "int4": multiplier = 0.3 # 4-bit + overhead
                
            runtime_gb = runtime_gb * multiplier
            
            # Conservative overhead (KV Cache, activations, etc)
            # For 7B models, 2GB-4GB overhead is typical depending on context length.
            overhead_gb = 2.0 if runtime_gb > 2.0 else 0.5
            
            final_estimate = runtime_gb + overhead_gb
            
            print(f"[DEBUG] Final Hub-based estimate for {model_id} ({target_dtype}): {final_estimate:.2f} GB")
            return final_estimate

        except Exception as e:
            print(f"[WARNING] Error predicting model size for {model_id}: {e}")
            return 16.0 # Safe fallback
