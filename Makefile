# A simple Makefile intended to facilitate repetitive tasks

# Dependencies installation
.PHONY: dep-install
dep-install:
	@echo "#### Installing dependencies ####"
	pip install numpy
	pip install scipy
	pip install scikit-learn
	# Used to run the examples
	pip install matplotlib
	# Package needed to run tests in an automate way
	pip install nose
	@echo "#### All dependencies were installed successfully ####"

# Test related commands
NOSETESTS ?= nosetests

.PHONY: test-all
test-all:
	nosetests

# Examples related commands
.PHONY: run-poisson
run-poisson:
	python examples/api/plot_poisson.py

.PHONY: run-poisson2
run-poisson2:
	python examples/api/plot_poisson2.py

# Documentation related commands
#   see doc/Makefile for more options
.PHONY: doc-build
doc-build:
	@echo "#### If there's any Exception, consider to run the command 'make doc-dependencies' ####"
	make -C doc html

OS := $(shell uname -s)
DOC := doc/_build/html/index.html
ifeq ($(OS), Linux)
	RUN := @xdg-open $(DOC)
else
	# OS X
	RUN := @open $(DOC)
endif

.PHONY: doc-run
doc-run:
	$(RUN)

# Sphinx Dependencies installation
.PHONY: doc-dependencies
doc-dependencies:
	pip install sphinx
	pip install sphinx-gallery

.PHONY: doc-clean
doc-clean:
	make -C doc clean

.PHONY: doc-publish
doc-publish:
	make -C doc install

.PHONY: install
install: dep-install
	@echo "#### Installing pyglmnet ####"
	python setup.py develop install
	@echo "#### Pyglmnet installed successfully ####"
	@echo "#### Access http://pavanramkumar.github.io/pyglmnet/index.html for more information and tutorials ####"
