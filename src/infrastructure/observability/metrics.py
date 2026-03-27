import socket
from prometheus_client import Counter, Gauge, Histogram, Summary

NODE_NAME = socket.gethostname()

# Bot activity metrics
MSGS_PROCESSED = Counter(
    "messages_processed_total",
    "Total messages processed",
    ["guanaco_name"],
)
THINK_DURATION = Summary(
    "think_duration_seconds",
    "Time spent processing messages",
    ["guanaco_name"],
)
BOT_WORKER_UP = Gauge(
    "bot_worker_up",
    "Whether a bot worker thread is running (1 = up, 0 = down)",
    ["bot_name"],
)
BOT_WORKER_LOOP_TOTAL = Counter(
    "bot_worker_loop_total",
    "Total work loop iterations by bot worker",
    ["bot_name"],
)
BOT_WORKER_ERRORS_TOTAL = Counter(
    "bot_worker_errors_total",
    "Total unhandled worker loop errors by bot worker",
    ["bot_name"],
)
BOT_UNREAD_MESSAGES = Gauge(
    "bot_unread_messages",
    "Unread messages detected in the last polling cycle",
    ["bot_name"],
)

# Model lifecycle and usage metrics
MODEL_SELECTION_TOTAL = Counter(
    "model_selection_total",
    "Total times a model was selected by routing logic",
    ["model_id", "complexity", "specialty"],
)
MODEL_LOAD_TOTAL = Counter(
    "model_load_total",
    "Model load attempts partitioned by status and location",
    ["model_id", "complexity", "device", "node", "status"],
)
MODEL_LOADED_INFO = Gauge(
    "model_loaded_info",
    "Loaded model placement marker (1 = loaded)",
    ["model_id", "complexity", "device", "node"],
)
MODEL_ACTIVE_REQUESTS = Gauge(
    "model_active_requests",
    "Active in-flight generation requests by model",
    ["model_id", "complexity"],
)
MODEL_CACHE_SIZE = Gauge(
    "model_cache_size",
    "Number of models currently cached in memory",
    ["node"],
)
MODEL_GENERATION_SECONDS = Histogram(
    "model_generation_seconds",
    "Model generation latency in seconds",
    ["model_id", "complexity", "status"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
MODEL_DOWNLOAD_EVENTS_TOTAL = Counter(
    "model_download_events_total",
    "Model download lifecycle events",
    ["model_id", "node", "status"],
)
MODEL_DOWNLOAD_STATUS_INFO = Gauge(
    "model_download_status_info",
    "Current model download status marker (1 = current status)",
    ["model_id", "node", "status"],
)
MODEL_DOWNLOAD_IN_PROGRESS = Gauge(
    "model_download_in_progress",
    "Whether the model is currently being downloaded in background (1 = in progress)",
    ["model_id", "node"],
)
MODEL_DOWNLOAD_PROGRESS_PERCENT = Gauge(
    "model_download_progress_percent",
    "Download progress percentage per model",
    ["model_id", "node"],
)
MODEL_DOWNLOAD_FILES_TOTAL = Gauge(
    "model_download_files_total",
    "Total files expected for model download",
    ["model_id", "node"],
)
MODEL_DOWNLOAD_FILES_COMPLETED = Gauge(
    "model_download_files_completed",
    "Downloaded files completed for model",
    ["model_id", "node"],
)
MODEL_DOWNLOAD_TOTAL_BYTES = Gauge(
    "model_download_total_bytes",
    "Total bytes expected for model download",
    ["model_id", "node"],
)
MODEL_DOWNLOAD_DOWNLOADED_BYTES = Gauge(
    "model_download_downloaded_bytes",
    "Downloaded bytes completed for model",
    ["model_id", "node"],
)
# Hardware utilization metrics
HARDWARE_CPU_USAGE_PERCENT = Gauge(
    "hardware_cpu_utilization_percent",
    "Current CPU utilization in percentage",
    ["node"],
)
HARDWARE_RAM_USED_BYTES = Gauge(
    "hardware_ram_used_bytes",
    "Current system RAM used in bytes",
    ["node"],
)
HARDWARE_RAM_TOTAL_BYTES = Gauge(
    "hardware_ram_total_bytes",
    "Total system RAM in bytes",
    ["node"],
)
HARDWARE_GPU_VRAM_USED_BYTES = Gauge(
    "hardware_gpu_vram_used_bytes",
    "Current GPU vRAM used in bytes",
    ["node", "gpu_id", "gpu_name"],
)
HARDWARE_GPU_VRAM_TOTAL_BYTES = Gauge(
    "hardware_gpu_vram_total_bytes",
    "Total GPU vRAM in bytes",
    ["node", "gpu_id", "gpu_name"],
)
HARDWARE_GPU_UTILIZATION_PERCENT = Gauge(
    "hardware_gpu_utilization_percent",
    "Current GPU utilization in percentage",
    ["node", "gpu_id", "gpu_name"],
)

# --- Router Metrics ---

# Recommended Alert Rule:
# - name: LowModelEffectiveness
#   expr: yaguarete_model_effectiveness < 0.6
#   for: 5m
#   labels:
#     severity: warning

ROUTER_MODEL_EFFECTIVENESS = Gauge(
    "yaguarete_model_effectiveness", 
    "Current effectiveness score of the model",
    ["model_id"]
)

ROUTER_AVG_TIME_PER_CHAR = Gauge(
    "yaguarete_avg_time_per_char_ms", 
    "Average processing time per input character in ms",
    ["model_id"]
)
