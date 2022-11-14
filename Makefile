# Makefile for Sphinx documentation
#

# Docs variables: You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build
APIDOCDIR     = source
PACKAGENAME   = f3dasm

# Build package variables
PACKAGEDIR    = dist

# Docs Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help doc-clean html apidoc latexpdf init init-dev test test-html build upload

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       	to make standalone documentation HTML files"
	@echo "  latexpdf   	to make documentation LaTeX files and run them through pdflatex"
	@echo "  init   	to install the requirements.txt"
	@echo "  init-dev   	to install the requirements_dev.txt"
	@echo "  test   	to run the tests with pytest"
	@echo "  test-html   	to run the tests with pytest and to create a HTML coverage report"
	@echo "  build		to build the package"
	@echo "  upload	to upload the package to the PyPi TEST index"

doc-clean:
	-rm -rf $(BUILDDIR)/*

html:
	cd docs/;
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	cd ..
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

apidoc:
	cd docs/;
	-rm -rf $(APIDOCDIR)/*
	sphinx-apidoc -o $(APIDOCDIR)/ ../$(PACKAGENAME)
	cd ..
	@echo
	@echo "Created API documentation using sphinx-apidoc. The source files are in $(APIDOCDIR)."


latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."


init:
	pip install -r requirements.txt
	
init-dev:
	pip install -r requirements_dev.txt

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
	pytest -v -s -m smoke --cov-report html
	@echo
	@echo "Smoke test finished. The coverage report HTML pages are in ./htmlcov/index.html"
	xdg-open ./htmlcov/index.html
	

test-html:
	pytest --cov-report html
	@echo
	@echo "Test finished. The coverage report HTML pages are in ./htmlcov/index.html"
	xdg-open ./htmlcov/index.html
	
build:
	@echo "Removing previous build"
	-rm -rf $(PACKAGEDIR)/*
	@echo "Building package"
	python setup.py sdist bdist_wheel --universal
	
upload-testpypi:
	make build
	@echo "Uploading the package to Test PyPI via Twine ..."
	twine upload -r testpypi $(PACKAGEDIR)/*

upload:
	make build
	@echo "Uploading the package to PyPI via Twine ..."
	twine upload -r pypi $(PACKAGEDIR)/*
