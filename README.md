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

Example data
------------

All of the following should work, with expression data. No guarantees on
splicing.

```
import flotilla
test_study = flotilla.embark('http://sauron.ucsd.edu/flotilla_projects/test_data/datapackage.json')
test_study.plot_pca()
test_study.interactive_pca()
test_study.plot_graph()
test_study.interactive_pca()
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

We will attempt to stick to the [`numpy` docstring specification](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) (aka
"`numpydoc`").

To make this easier, I use ["Live Templates" in PyCharm]
(http://peter-hoffmann.com/2010/python-live-templates-for-pycharm.html),
check out the instructions [here](https://github
.com/YeoLab/PyCharm-Python-Templates) for how to install and use them.

What flotilla is not
====================

Flotilla does not claim to solve the data management problem of biology,
i.e. how you store all the data associated with a particular study that was
investigating a specific biological question. Flotilla only makes it easy to
integrate all those data parts together.