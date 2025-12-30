.PHONY: help install test test-cov check docs docs-serve clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies with uv"
	@echo "  make test        - Run tests with pytest"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make check       - Run all pre-commit hooks"
	@echo "  make docs        - Build documentation"
	@echo "  make docs-serve  - Serve documentation locally"
	@echo "  make clean       - Remove build artifacts"

install:
	uv sync --all-extras

test:
	uv run python -m pytest tests/ -v

test-cov:
	uv run python -m pytest tests/ -v --cov=src/lonpy --cov-report=term-missing --cov-report=html

check:
	pre-commit run --all-files

docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf site/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
