# 🐳 Yaguarete LLM Proxy - Guía Docker

## Resumen de cambios

✅ **Dependencias actualizadas** - `requirements.txt` con versiones pinned compatibles  
✅ **Dockerfile creado** - Multi-stage build optimizado con soporte CUDA  
✅ **Docker Compose** - Configuración completa con GPU y Phoenix  
✅ **Script de utilidad** - `docker-build.sh` para gestionar la imagen  

## Versiones compatibles instaladas

- `transformers==4.36.2`
- `torch==2.1.1`
- `accelerate==0.25.0`
- `bitsandbytes>=0.41.0`
- Todas las demás dependencias pinned

## Inicio rápido

### Opción 1: Con Docker Compose (Recomendado ⭐)

```bash
cd /home/jose/projects/metahumans/yaguarete

# Build e inicia todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f yaguarete
```

### Opción 2: Con script automatizado

```bash
cd /home/jose/projects/metahumans/yaguarete

# Build
./docker-build.sh build

# Iniciar
./docker-build.sh compose-up

# Ver logs
./docker-build.sh logs
```

### Opción 3: Docker manual (sin Compose)

```bash
cd /home/jose/projects/metahumans/yaguarete

# Build
docker build -t yaguarete:latest .

# Run con GPU
docker run -d \
  --name yaguarete \
  --gpus all \
  -p 8001:8001 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /tmp/donmingo-offload:/tmp/donmingo-offload \
  yaguarete:latest

# Ver logs
docker logs -f yaguarete
```

## Verificar que funciona

```bash
# Health check
curl http://localhost:8001/health

# Listar modelos disponibles
curl http://localhost:8001/v1/models

# Ver métricas Prometheus
curl http://localhost:8001/metrics
```

## Estructura del Dockerfile

El Dockerfile usa **multi-stage build** para optimizar el tamaño:

1. **Builder stage**: Compila bitsandbytes y todas las dependencias
2. **Final stage**: Solo copia lo compilado (reduce tamaño ~2GB)

## Variables de entorno disponibles

En `docker-compose.yml` puedes ajustar:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1      # GPUs a usar
  - PYTHONUNBUFFERED=1             # Output inmediato
```

## Volúmenes

- `./data`: Cache de modelos y datos
- `./logs`: Logs de la aplicación
- `/tmp/donmingo-offload`: Offload de memoria GPU

## Detener servicios

```bash
# Con Docker Compose
docker-compose down

# O con el script
./docker-build.sh compose-down

# O manual
docker stop yaguarete
docker rm yaguarete
```

## Troubleshooting

### GPU no detectada
```bash
# Verificar que NVIDIA runtime existe
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime nvidia-smi

# Asegurar que /etc/docker/daemon.json tiene:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Error de memoria
- Los modelos 7B requieren ~6-8GB VRAM
- Se usa offloading automático a `/tmp/donmingo-offload`
- Asegurar que hay espacio en disco: `df -h /tmp/`

### Build falla en bitsandbytes
```bash
# Reinstalar drivers NVIDIA
nvidia-smi

# O buildear sin CUDA (más lento):
docker build --build-arg CUDA_SUPPORT=0 -t yaguarete:latest .
```

## Archivos creados/modificados

- ✏️ `requirements.txt` - Versiones pinned
- 🆕 `Dockerfile` - Multi-stage build
- 🆕 `docker-compose.yml` - Orquestación con Phoenix
- 🆕 `.dockerignore` - Archivos a excluir del build
- 🆕 `docker-build.sh` - Script de utilidad
- 🆕 `DOCKER_SETUP.md` - Guía en inglés

## Siguiente paso

Ejecuta ahora:

```bash
cd /home/jose/projects/metahumans/yaguarete
docker-compose build
docker-compose up -d
docker-compose logs -f yaguarete
```

¡Listo! La aplicación estará disponible en `http://localhost:8001` 🚀
