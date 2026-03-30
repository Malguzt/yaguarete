.PHONY: run install setup-venv clean test

VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)
	@echo "Ejecutando Yaguarete..."
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
	@echo "Limpieza completada."
