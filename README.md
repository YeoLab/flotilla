[![Build Status](https://travis-ci.org/YeoLab/flotilla.svg?branch=modalities)](https://travis-ci.org/YeoLab/flotilla)[![Coverage Status](https://coveralls.io/repos/YeoLab/flotilla/badge.png)](https://coveralls.io/r/YeoLab/flotilla)

flotilla
========
![flotilla Logo](flotilla.png)

Download
========

```
git clone https://github.com/YeoLab/flotilla.git
```

Install
=======

for some reason patsy doesn't always automatically with pip, use easy_install first instead

```
easy_install -U patsy
cd flotilla
pip install .
```


Example data
------------

All of the following should work, with expression data. No guarantees on
splicing.


```
import flotilla
test_study = flotilla.embark('http://sauron.ucsd.edu/flotilla_projects/neural_diff_chr22/datapackage.json')
test_study.plot_pca()
test_study.interactive_pca()
test_study.plot_graph()
test_study.interactive_graph()
test_study.plot_classifier()
test_study.interactive_classifier()
```


For developers
==============

Please put ALL import statements at the top of the `*.py` file (potentially underneath docstrings, of course).
The only exception is if a package is not listed in `requirements.txt`,then a "function-only" import may be allowed.
If this doesn't make sense to you, just put the import at the top of the file.



Naming conventions
------------------

When in doubt, please defer to [Python Enhancement Proposal 8 (aka PEP8))
[http://legacy.python.org/dev/peps/pep-0008/] and the [Zen of Python]
(http://legacy.python.org/dev/peps/pep-0020/)

* Classes are `CamelCase`, e.g.:  `BaseData` and `PCAViz`
* Functions are `lower_case_with_underscores`, e.g. `go_enrichment` and
`binify`
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


Testing
=======