# A simple Makefile intended to facilitate repetitive tasks

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
NOSETESTS_OPTIONS := $(shell pip list | grep nose-timer > /dev/null && \
                       echo '--with-timer --timer-top-n 50')

# Dependencies installation
.PHONY: dependencies
dependencies:
	@echo "#### Installing dependencies ####"
	pip install numpy
	pip install scipy
	@echo "#### All dependencies were installed successfully ####"

# Documentation related commands
#   see doc/Makefile for more options
.PHONY: doc
doc:
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
install: dependencies
	@echo "#### Installing pyglmnet ####"
	python setup.py develop install
	@echo "#### Pyglmnet installed successfully ####"
	@echo "#### Access http://pavanramkumar.github.io/pyglmnet/index.html for more information and tutorials ####"

.PHONY: all
all: install doc-dependencies doc-build

all: clean test install doc-dependencies doc

.PHOMY: test
test:

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean: clean-build clean-pyc clean-so doc-clean

test-code:
	$(NOSETESTS) -s pyglmnet $(NOSETESTS_OPTIONS)
test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture `find doc/ -name '*.rst'`

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=pyglmnet pyglmnet

test: test-code test-doc
