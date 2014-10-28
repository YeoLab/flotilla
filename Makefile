export SHELL := /bin/bash

test:
	py.test

coverage:
	py.test --cov flotilla flotilla/test/

lint:
	pyflakes -x W flotilla flotilla/test
	pep8 flotilla flotilla/test
