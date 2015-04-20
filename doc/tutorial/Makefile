#!/bin/bash

dir=.
ipynbs=$(wildcard $(dir:=/*.ipynb))

notebooks:

	echo $(ipynbs)
	$(foreach ipynb, $(ipynbs), tools/nb_to_doc.py $(echo $(ipynb) | sed 's:.ipynb::');)
