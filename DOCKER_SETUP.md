# Yaguarete LLM Proxy - Docker Setup

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- GPU with at least 6GB VRAM recommended

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)

Build and run with all dependencies automatically installed:

```bash
docker-compose build
docker-compose up -d
```

### Option 2: Manual Docker Build

```bash
docker build -t yaguarete:latest .
```

## Running the Container

### With Docker Compose

```bash
# Start all services (Yaguarete + Phoenix for observability)
docker-compose up -d

# View logs
docker-compose logs -f yaguarete

# Stop services
docker-compose down
```

### With Raw Docker

```bash
# Build
docker build -t yaguarete:latest .

# Run with GPU support
docker run -d \
  --name yaguarete-llm-proxy \
  --gpus all \
  -p 8001:8001 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /tmp/donmingo-offload:/tmp/donmingo-offload \
  yaguarete:latest

# View logs
docker logs -f yaguarete-llm-proxy

# Stop
docker stop yaguarete-llm-proxy
```

## Verification

Once running, verify the service:

```bash
# Check health
curl http://localhost:8001/health

# List available models
curl http://localhost:8001/v1/models

# View metrics
curl http://localhost:8001/metrics
```

## Environment Variables

Configure these in `docker-compose.yml`:

- `CUDA_VISIBLE_DEVICES`: GPU IDs to use (default: 0,1)
- `PYTHONUNBUFFERED`: Set to 1 for immediate log output

## Volumes

- `./data`: Model cache and data storage
- `./logs`: Application logs
- `/tmp/donmingo-offload`: GPU memory offload directory

## Troubleshooting

### GPU not detected
```bash
# Check if NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# Ensure /etc/docker/daemon.json has:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Memory issues
- Reduce model size via `ModelCatalog`
- Increase GPU memory with larger offload folder
- Check `/tmp/donmingo-offload` disk space

### Build fails on bitsandbytes
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Update Docker to latest version
- Try building without GPU: `docker build --build-arg CUDA_SUPPORT=0 -t yaguarete:latest .`
