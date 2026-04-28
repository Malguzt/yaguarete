.PHONY: run run-local install setup-venv clean test docker-build docker-up docker-down docker-logs docker-rebuild docker-clean build up down logs sonar sonar-list sonar-issues sonar-keys sonar-top sonar-detail sonar-issue-detail

VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
DOCKER = docker compose

# Load .env file
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

# ============== SonarQube ==============
SONAR_NETWORK = sonarqube-local_sonarnet
SONAR_TOKEN ?= $(SONAR_TOKEN)
SONAR_URL ?= $(SONAR_URL)

# ============== VENV / Local Development ==============

run:
	@echo "Ejecutando Yaguarete en Docker..."
	@$(DOCKER) up --build

run-local: $(VENV)
	@echo "Ejecutando Yaguarete localmente..."
	@$(PYTHON) src/main.py

install: $(VENV)
	@echo "Instalando dependencias desde requirements.txt..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

$(VENV):
	@echo "Creando entorno virtual (.venv)..."
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

test: $(VENV)
	@echo "Ejecutando tests con pytest..."
	@$(PYTHON) -m pytest

clean:
	@echo "Limpiando archivos temporales y entorno virtual..."
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@$(DOCKER) down 2>/dev/null || true
	@echo "Limpieza completada."

# ============== Docker ==============

docker-build:
	@echo "🐳 Construyendo imagen Docker..."
	@$(DOCKER) build --no-cache

docker-up:
	@echo "🚀 Iniciando servicios..."
	@$(DOCKER) up -d
	@echo "✅ Servicios iniciados. Usa: make docker-logs"

docker-down:
	@echo "⏹️  Deteniendo servicios..."
	@$(DOCKER) down
	@echo "✅ Servicios detenidos."

docker-logs:
	@$(DOCKER) logs -f yaguarete

docker-rebuild: docker-down docker-build docker-up
	@echo "✅ Reconstrucción completada."

docker-clean:
	@echo "🧹 Limpiando Docker..."
	@$(DOCKER) down -v
	@docker system prune -f
	@echo "✅ Docker limpio."

# ============== SonarQube Commands ==============

sonar:
	@echo "📡 Ejecutando SonarQube Scanner..."
	@docker run --rm \
		--network $(SONAR_NETWORK) \
		-v "$(shell pwd):/usr/src" \
		sonarsource/sonar-scanner-cli \
		-Dsonar.host.url=$(SONAR_URL) \
		-Dsonar.login=$(SONAR_TOKEN)

sonar-list: sonar-issues

sonar-issues:
	@$(PYTHON) scripts/sonar_issues.py --list

sonar-keys:
	@$(PYTHON) scripts/sonar_issues.py --keys

sonar-top:
	@if [ -z "$(N)" ]; then \
		echo "❌ Error: Debes proporcionar N."; \
		echo "Uso: make sonar-top N=3"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/sonar_issues.py --top $(N)

sonar-detail: sonar-issue-detail

sonar-issue-detail:
	@if [ -z "$(KEY)" ]; then \
		echo "❌ Error: Debes proporcionar un KEY."; \
		echo "Uso: make sonar-issue-detail KEY=id_del_issue"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/sonar_issues.py --detail $(KEY)

# ============== Aliases ==============

build: sonar docker-build
	@echo "✅ Build y análisis de Sonar completado."

up: docker-up
down: docker-down
logs: docker-logs
rebuild: docker-rebuild

# ============== Help ==============

help:
	@echo "════════════════════════════════════════"
	@echo "  Yaguarete Makefile Commands"
	@echo "════════════════════════════════════════"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make run                - Ejecutar stack en Docker (foreground)"
	@echo "  make build              - Construir imagen Docker"
	@echo "  make up                 - Iniciar servicios (docker-compose up -d)"
	@echo "  make down               - Detener servicios"
	@echo "  make logs               - Ver logs en tiempo real"
	@echo "  make rebuild            - Reconstruir e iniciar"
	@echo "  make docker-clean       - Limpiar volúmenes y prune"
	@echo ""
	@echo "💻 Local Development:"
	@echo "  make run-local          - Ejecutar app localmente (.venv)"
	@echo "  make install            - Instalar dependencias en venv"
	@echo "  make test               - Correr tests"
	@echo "  make clean              - Limpiar venv y temporales"
	@echo ""
	@echo "📊 SonarQube:"
	@echo "  make sonar              - Ejecutar análisis de SonarQube"
	@echo "  make sonar-issues       - Listar issues resumidos"
	@echo "  make sonar-keys         - Listar todas las claves de issues"
	@echo "  make sonar-top N=3      - Mostrar los TOP N issues y sus claves"
	@echo "  make sonar-issue-detail - Ver detalle de un issue (KEY=...)"
	@echo ""
	@echo "Quick start:"
	@echo "  make run"
	@echo ""
