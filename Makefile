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
	pyflakes -X flotilla/__init__.py -X flotilla/data_model/__init__.py -X flotilla/visualize/__init__.py flotilla  flotilla/compute flotilla/visualize flotilla/data_model flotilla/test

pep8:
	pep8 flotilla flotilla/test
