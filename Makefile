# FormCoach — every workflow has a target here. Each target delegates to `uv run formcoach ...`
# so teammates without `make` (plain PowerShell on Windows) can run the same commands directly.
# Run `make` or `make help` to list targets.
.DEFAULT_GOAL := help
SHELL := /bin/bash

UV      ?= uv
RUN     := $(UV) run
PIO     := $(UV) tool run platformio
DATASET ?= mmfit
SOURCE  ?= replay
PORT    ?=
MODEL   ?= cnn

.PHONY: help setup setup-all lint format test data features train-gate eval demo record \
	    fw-build fw-upload fw-monitor clean

help: ## list targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create .venv, install formcoach + dev tools, install pre-commit hooks
	$(UV) sync --group dev
	$(RUN) pre-commit install
	@echo "Setup complete. Next: make demo"

setup-all: ## like setup, plus vision/device/train/app extras (large downloads)
	$(UV) sync --all-extras --group dev
	$(RUN) pre-commit install

lint: ## ruff check + format check (what CI runs)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## auto-format and auto-fix lint
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test: ## run the pytest suite
	$(RUN) pytest

data: ## fetch a public dataset into data/external/ (DATASET=mmfit|recofit|recgym)
	$(RUN) formcoach data fetch --dataset $(DATASET)

features: ## build windows + features into data/processed/
	$(RUN) formcoach features build

train-gate: ## train the gate model (MODEL=rf|cnn) and export int8
	$(RUN) formcoach train gate --model $(MODEL)

eval: ## regenerate every table and figure under reports/
	$(RUN) formcoach eval all

demo: ## run the full pipeline (SOURCE=replay|serial|ble; replay needs no hardware)
	$(RUN) formcoach demo --source $(SOURCE) $(if $(PORT),--port $(PORT),)

record: ## record a team session (SOURCE=serial|ble PORT=/dev/ttyACM0|COM5)
	$(RUN) formcoach record --source $(SOURCE) $(if $(PORT),--port $(PORT),)

fw-build: ## compile the firmware (PlatformIO via uvx; also runs in CI)
	$(PIO) run -d firmware

fw-upload: ## flash the firmware over USB-C
	$(PIO) run -d firmware -t upload

fw-monitor: ## open the serial monitor (115200 baud)
	$(PIO) device monitor -d firmware

clean: ## remove caches and build artifacts (never touches data/)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage firmware/.pio
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
