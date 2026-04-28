#!/bin/bash
# Build and manage Yaguarete Docker image

set -e

REGISTRY="${REGISTRY:-}"
IMAGE_NAME="${REGISTRY}yaguarete"
IMAGE_TAG="${TAG:-latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-.}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🐳 Yaguarete Docker Build Manager${NC}"
echo "=================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Dockerfile: $DOCKERFILE_PATH/Dockerfile"
echo ""

case "${1:-build}" in
  build)
    echo -e "${YELLOW}📦 Building Docker image...${NC}"
    docker build \
      -t "$IMAGE_NAME:$IMAGE_TAG" \
      -t "$IMAGE_NAME:latest" \
      -f "$DOCKERFILE_PATH/Dockerfile" \
      "$DOCKERFILE_PATH"
    echo -e "${GREEN}✅ Build complete!${NC}"
    echo ""
    echo "Run with Docker Compose:"
    echo "  docker-compose up -d"
    echo ""
    echo "Or manually:"
    echo "  docker run --gpus all -p 8001:8001 $IMAGE_NAME:$IMAGE_TAG"
    ;;
  
  push)
    if [ -z "$REGISTRY" ]; then
      echo -e "${RED}❌ REGISTRY not set. Cannot push.${NC}"
      exit 1
    fi
    echo -e "${YELLOW}📤 Pushing image to registry...${NC}"
    docker push "$IMAGE_NAME:$IMAGE_TAG"
    docker push "$IMAGE_NAME:latest"
    echo -e "${GREEN}✅ Push complete!${NC}"
    ;;
  
  run)
    echo -e "${YELLOW}🚀 Starting Yaguarete container...${NC}"
    docker run -d \
      --name yaguarete-llm-proxy \
      --gpus all \
      -p 8001:8001 \
      -e CUDA_VISIBLE_DEVICES=0,1 \
      -v /tmp/donmingo-offload:/tmp/donmingo-offload \
      "$IMAGE_NAME:$IMAGE_TAG"
    echo -e "${GREEN}✅ Container started!${NC}"
    echo "Monitor logs: docker logs -f yaguarete-llm-proxy"
    ;;
  
  stop)
    echo -e "${YELLOW}⏹️  Stopping Yaguarete container...${NC}"
    docker stop yaguarete-llm-proxy 2>/dev/null || echo "Container not running"
    docker remove yaguarete-llm-proxy 2>/dev/null || echo "Container not found"
    echo -e "${GREEN}✅ Stopped!${NC}"
    ;;
  
  compose-build)
    echo -e "${YELLOW}📦 Building with Docker Compose...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}✅ Build complete!${NC}"
    ;;
  
  compose-up)
    echo -e "${YELLOW}🚀 Starting services with Docker Compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Services started!${NC}"
    echo "Monitor: docker-compose logs -f"
    ;;
  
  compose-down)
    echo -e "${YELLOW}⏹️  Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ Stopped!${NC}"
    ;;
  
  logs)
    docker logs -f yaguarete-llm-proxy
    ;;
  
  clean)
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    docker stop yaguarete-llm-proxy 2>/dev/null || true
    docker rm yaguarete-llm-proxy 2>/dev/null || true
    docker rmi "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null || true
    echo -e "${GREEN}✅ Clean!${NC}"
    ;;
  
  *)
    echo -e "${RED}Unknown command: $1${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  push               Push image to registry (requires REGISTRY env var)"
    echo "  run                Run container manually"
    echo "  stop               Stop running container"
    echo "  compose-build      Build with Docker Compose"
    echo "  compose-up         Start services with Docker Compose"
    echo "  compose-down       Stop services with Docker Compose"
    echo "  logs               View container logs"
    echo "  clean              Remove image and containers"
    exit 1
    ;;
esac
