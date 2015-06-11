export SHELL := /bin/bash

test:
	cp testing/matplotlibrc .
	py.test
	rm matplotlibrc

coverage:
	cp testing/matplotlibrc .
	coverage run --source flotilla --omit=test --module py.test -v
	rm matplotlibrc

lint:
	flake8 --exclude flotilla/external,doc flotilla
