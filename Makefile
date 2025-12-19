.DEFAULT_GOAL := help

PACKAGEDIR := dist
COVERAGEREPORTDIR := coverage_html_report

.PHONY: help init init-dev test test-smoke test-smoke-html test-html build upload upload-testpypi docs lint

help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  init                Install the requirements.txt"
	@echo "  init-dev            Install the requirements_dev.txt"
	@echo "  test                Run the tests with pytest"
	@echo "  test-html           Run the tests with pytest and open the HTML coverage report"
	@echo "  test-smoke          Run the smoke tests with pytest"
	@echo "  test-smoke-html     Run the smoke tests with pytest and open the HTML coverage report"
	@echo "  build               Build the package"
	@echo "  upload              Upload the package to the PyPi index"
	@echo "  upload-testpypi     Upload the package to the PyPi-test index"
	@echo "  docs                Build the documentation with mkdocs"
	@echo "  lint                Lint the code with ruff"

init:
	pip install -r requirements.txt

init-dev:
	pip install -r requirements_dev

test:
	uv run pytest -v -s -m smoke

test-smoke-html:
	pytest -v -s -m smoke
	xdg-open ./$(COVERAGEREPORTDIR)/index.html

test-html:
	pytest
	xdg-open ./$(COVERAGEREPORTDIR)/index.html

build:
	uv build

upload-testpypi:
	$(MAKE) build
	twine upload -r testpypi $(PACKAGEDIR)/* --verbose

upload:
	$(MAKE) build
	twine upload $(PACKAGEDIR)/* --verbose

docs:
	mkdocs build

lint:
	ruff check
