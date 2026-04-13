.PHONY: help clean install build test lint format docs visualize

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in development mode"
	@echo "  make build         - Build Cython extensions"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make visualize     - Generate example visualizations"
	@echo "  make docs          - Generate documentation"

install:
	pip install -e ".[dev]"

build:
	python setup.py build_ext --inplace

test:
	pytest tests/ -v --cov=cfptree --cov-report=html

lint:
	flake8 cfptree tests examples
	black --check cfptree tests examples

format:
	black cfptree tests examples

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/

visualize:
	python examples/example_usage.py

docs:
	cd docs && make html