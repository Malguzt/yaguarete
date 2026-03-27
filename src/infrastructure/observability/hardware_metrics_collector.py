import threading
import time
import psutil
import torch
from infrastructure.observability.metrics import (
    HARDWARE_CPU_USAGE_PERCENT,
    HARDWARE_RAM_USED_BYTES,
    HARDWARE_RAM_TOTAL_BYTES,
    HARDWARE_GPU_VRAM_USED_BYTES,
    HARDWARE_GPU_VRAM_TOTAL_BYTES,
    HARDWARE_GPU_UTILIZATION_PERCENT,
    NODE_NAME,
)

class HardwareMetricsCollector:
    """Background collector for hardware metrics."""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._collect_loop, daemon=True, name="HardwareMetricsCollector")
        self._thread.start()
        print("[INFO] Hardware metrics collector started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _collect_loop(self):
        while not self._stop_event.is_set():
            try:
                # CPU & RAM
                HARDWARE_CPU_USAGE_PERCENT.labels(node=NODE_NAME).set(psutil.cpu_percent())
                vm = psutil.virtual_memory()
                HARDWARE_RAM_USED_BYTES.labels(node=NODE_NAME).set(vm.used)
                HARDWARE_RAM_TOTAL_BYTES.labels(node=NODE_NAME).set(vm.total)

                # GPU
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        name = torch.cuda.get_device_name(i)
                        total_vram = torch.cuda.get_device_properties(i).total_memory
                        try:
                            free_vram, _ = torch.cuda.mem_get_info(i)
                            used_vram = total_vram - free_vram
                        except Exception:
                            used_vram = torch.cuda.memory_allocated(i)
                        
                        HARDWARE_GPU_VRAM_USED_BYTES.labels(node=NODE_NAME, gpu_id=i, gpu_name=name).set(used_vram)
                        HARDWARE_GPU_VRAM_TOTAL_BYTES.labels(node=NODE_NAME, gpu_id=i, gpu_name=name).set(total_vram)
                        
                        # Note: Simple utilization % estimation if nvidia-ml-py is not available
                        # In a real environment, pynvml is better for utilization %
                        util_percent = (used_vram / total_vram) * 100 if total_vram > 0 else 0
                        HARDWARE_GPU_UTILIZATION_PERCENT.labels(node=NODE_NAME, gpu_id=i, gpu_name=name).set(util_percent)

            except Exception as e:
                print(f"[WARNING] Error collecting hardware metrics: {e}")
            
            time.sleep(self.interval)
