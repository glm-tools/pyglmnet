# A simple Makefile intended to facilitate repetitive tasks

# Dependencies installation
.PHONY: dep-install
dep-install:
	@echo "#### Installing dependencies ####"
	pip install numpy
	pip install scipy
	@echo "#### All dependencies were installed successfully ####"

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
	pip install matplotlib

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

.PHONY: all
all: install doc-dependencies doc-build

.PHONY: clean
clean:

.PHOMY: test
test:
