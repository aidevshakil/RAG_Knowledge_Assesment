PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip

install: $(BIN)/python ## Create venv and install everything
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e ".[dev]"

install-min: $(BIN)/python ## Minimal offline install (numpy store + extractive LLM)
	$(BIN)/pip install -e ".[dev]"

run: ## Launch the Streamlit UI
	$(BIN)/streamlit run app/main.py

ingest: ## Ingest files/folders from the CLI: make ingest SRC="docs/ some.pdf"
	$(BIN)/python -m rag_assistant.cli ingest $(SRC)

ask: ## One-off query: make ask Q="what is RAG?"
	$(BIN)/python -m rag_assistant.cli ask "$(Q)"

bench: ## Run the benchmark suite: make bench SUITE=eval/questions.json
	$(BIN)/python -m rag_assistant.cli benchmark --suite $(SUITE)

test: ## Run unit tests
	$(BIN)/pytest

lint: ## Lint + format check
	$(BIN)/ruff check src app tests
	$(BIN)/ruff format --check src app tests

fmt: ## Auto-format
	$(BIN)/ruff format src app tests
	$(BIN)/ruff check --fix src app tests

clean: ## Remove caches (keeps data/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

.PHONY: help install install-min run ingest ask bench test lint fmt clean
