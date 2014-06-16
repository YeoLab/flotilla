flotilla
========
download with:
```
git clone --recursive https://github.com/YeoLab/flotilla.git
```
and download the singlecell project (for testing porpoises) with:
```
git clone https://github.com/YeoLab/neural_diff_project.git
```

build/install with:

note: for some reason patsy isn't installing automatically with pip, use easy_install first instead
```
easy_install -U patsy
cd flotilla
pip install .
cd ..
cd neural_diff_project
pip install -e .
cd ..
```
start a notebook
```
serve_flotilla_notebook neural_diff_project/notebook
```



check intro to flotila.html/ipynb for instructions


How to make a new flotilla project:

from flotilla copy barebones_project/ into a new directory

```
cp -r ./barebones_project ../new_project
```
rename the directory inside barebones_project to your new project name

```
mv ../new_project/barebones_project ../new_project/new_project
```



For developers
==============

Please put ALL import statements at the top of the `*.py` file. The only
exception is if a package is not listed in `requirements.txt`,
then a "function-only" import may be allowed. If this doesn't make sense to
you, just put the import at the top of the file.

Naming conventions
------------------

When in doubt, please defer to [Python Enhancement Proposal 8 (aka PEP8))
[http://legacy.python.org/dev/peps/pep-0008/] and the [Zen of Python]
(http://legacy.python.org/dev/peps/pep-0020/)

* Classes are `CamelCase`, e.g.:  `BaseData` and `PCAViz`
* Functions are `lower_case_with_underscores`, e.g. `go_enrichment` and
`binify`
* If there is a choice between singular and plural, go with the singular form
 of the word
* Explicit is better than implicit

Docstring conventions
---------------------

We will attempt to stick to the `numpy` docstring specification (aka
"`numpydoc`"), described here: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

To make this easier, I use ["Live Templates" in PyCharm]
(http://peter-hoffmann.com/2010/python-live-templates-for-pycharm.html),
check out the instructions [here](https://github
.com/YeoLab/PyCharm-Python-Templates) for how to install and use them.
