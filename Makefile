clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/


test: clean-test
	-python -m coverage run --source=changepointmodel/ -m pytest -s -v && coverage report -m --fail-under=90
	mypy changepointmodel --strict

test-single-module: clean-test
	pytest $(module) -v -s

install: clean 
	poetry install 