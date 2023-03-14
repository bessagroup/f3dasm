# Makefile for f3dasm development

# Build package variables
PACKAGEDIR    = dist
COVERAGEREPORTDIR = coverage_html_report

.PHONY: help init init-dev test test-smoke test-smoke-html test-html build upload upload-testpypi

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  init   			to install the requirements.txt"
	@echo "  init-dev   			to install the requirements_dev.txt"
	@echo "  test   			to run the tests with pytest"
	@echo "  test-html   			to run the tests with pytest and to open the HTML coverage report"
	@echo "  test-smoke   			to run the smoke tests with pytes"
	@echo "  test-smoke-html   		to run the smoke tests with pytest and to open the HTML coverage report"
	@echo "  build				to build the package"
	@echo "  upload			to upload the package to the PyPi index"
	@echo "  upload-testpypi		to upload the package to the PyPi-test index"

init:
	pip install -r requirements.txt

init-dev:
	pip install -r requirements_dev
	
test:
	pytest
	@echo
	@echo "Test finished"

test-smoke:
	pytest -v -s -m smoke
	@echo
	@echo "Smoke test finished"

test-leo:
	pytest -v -s -m leo
	@echo
	@echo "Leo test finished"
	
test-smoke-html:
	pytest -v -s -m smoke
	@echo
	@echo "Smoke test finished. The coverage report HTML pages are in ./$(COVERAGEREPORTDIR)/index.html"
	xdg-open ./$(COVERAGEREPORTDIR)/index.html
	

test-html:
	pytest
	@echo
	@echo "Test finished. The coverage report HTML pages are in ./$(COVERAGEREPORTDIR)/index.html"
	xdg-open ./$(COVERAGEREPORTDIR)/index.html
	
build:
	@echo "Removing previous build"
	-rm -rf $(PACKAGEDIR)/*
	@echo "Building package"
	python -m build
	
upload-testpypi:
	make build
	@echo "Uploading the package to Test PyPI via Twine ..."
	twine upload -r testpypi $(PACKAGEDIR)/* --verbose

upload:
	@echo "Uploading the package to PyPI via Twine ..."
	twine upload $(PACKAGEDIR)/* --verbose