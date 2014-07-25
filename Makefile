export SHELL := /bin/bash

tests:
	py.test

coverage:
	py.test --cov flotilla test/

lint:
	pyflakes -x W flotilla
	pep8 flotilla
