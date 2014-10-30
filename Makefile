export SHELL := /bin/bash

test:
	py.test

coverage:
	py.test --cov flotilla --cov-report term-missing flotilla/test/

lint:
	pyflakes -x W flotilla flotilla/test
	pep8 flotilla flotilla/test
