.PHONY: help setup test test-unit test-property test-integration coverage clean format lint install

help:
	@echo "LLM Judge Auditor - Development Commands"
	@echo "========================================"
	@echo "setup              - Create virtual environment and install dependencies"
	@echo "install            - Install package in editable mode"
	@echo "test               - Run all tests"
	@echo "test-unit          - Run unit tests only"
	@echo "test-property      - Run property-based tests only"
	@echo "test-integration   - Run integration tests only"
	@echo "coverage           - Run tests with coverage report"
	@echo "format             - Format code with black"
	@echo "lint               - Check code with ruff"
	@echo "clean              - Remove build artifacts and cache"

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv venv
	@echo "Activating and installing dependencies..."
	@echo "Run: source venv/bin/activate (or venv\\Scripts\\activate.bat on Windows)"
	@echo "Then run: make install"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

test-unit:
	pytest tests/unit/

test-property:
	pytest tests/property/

test-integration:
	pytest tests/integration/

coverage:
	pytest --cov=llm_judge_auditor --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

format:
	black src tests examples

lint:
	ruff check src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
