#!/bin/bash

dir=.
ipynbs=$(wildcard $(dir:=/*.ipynb))

notebooks:

	echo $(ipynbs)
	$(foreach ipynb, $(ipynbs), ipython nbconvert --to rst "$(ipynb)";)
