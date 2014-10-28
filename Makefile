export SHELL := /bin/bash

test:
	py.test

coverage:
	py.test --cov flotilla test/

lint:
	pyflakes -x W flotilla test
	pep8 flotilla test
