.PHONY: test lint typecheck format all

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

all: lint typecheck test
