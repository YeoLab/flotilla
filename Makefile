export SHELL := /bin/bash

test:
	cp testing/matplotlibrc .
	py.test
	rm matplotlibrc

coverage:
	cp testing/matplotlibrc .
	py.test --durations=20 --cov flotilla --cov-report term-missing flotilla/test/
	rm matplotlibrc

lint:
	pyflakes -x W flotilla flotilla/test

pep8:
	pep8 flotilla flotilla/test
