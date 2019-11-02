# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
PYTESTS ?= pytest

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

inplace:
	$(PYTHON) setup.py install

test: inplace check-manifest
	rm -f .coverage
	$(PYTESTS) pyglmnet

test-doc:
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors

test-coverage:
	rm -rf coverage .coverage
	$(PYTESTS) --cov=pyglmnet --cov-report html:coverage

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

upload-pipy:
	python setup.py sdist bdist_egg register upload

check-manifest:
	check-manifest --ignore .circleci*,doc,.DS_Store,paper

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count pyglmnet examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

pep:
	@$(MAKE) -k flake pydocstyle

build-doc:
	cd doc; make clean
	cd doc; make html
