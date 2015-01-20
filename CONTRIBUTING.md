
For developers
==============

Please put ALL import statements at the top of the `*.py` file (potentially underneath docstrings, of course).
The only exception is if a package is not listed in `requirements.txt`,then a "function-only" import may be allowed.
If this doesn't make sense to you, just put the import at the top of the file.


Install in development mode
---------------------------

First clone the repository,

    git clone git@github.com:YeoLab/flotilla

Change directories to the flotilla directory you just made,

    cd flotilla

Now install via `pip` in "editable" mode, aka "develop" mode:

    pip install -e .

Git branching
-------------

We use the [GitHub-Flow](http://scottchacon.com/2011/08/31/github-flow.html) model.

To contribute:

1.    Make a branch off of the master.

2.    Commit updates exclusively to your branch.

3.    When changes on that branch are finished, open a [pull request](https://help.github.com/articles/using-pull-requests/).

4.    Once someone has reviewed and approved of the changes on your branch, you should immediately merge your branch into the master.

Naming conventions
------------------

When in doubt, please defer to [Python Enhancement Proposal 8 (aka [PEP8]
(http://legacy.python.org/dev/peps/pep-0008/)) and the [Zen of Python]
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
check out the instructions [here](https://github.com/YeoLab/PyCharm-Python-Templates) for how to install and use them.

Testing
-------

In the source directory (wherever you cloned `flotilla` to that has this README.md file), do:

    make test

This will run the unit test suite.

### Coverage

To check coverage of the test suite, run

    make coverage

in the source directory.


PEP8 Conventions
----------------

To run `pep8` and `pyflakes` over the code, make sure you have [this fork]
(pip install https://github.com/dcramer/pyflakes/tarball/master) of
`pyflakes` installed (e.g. via `pip install https://github
.com/dcramer/pyflakes/tarball/master`) and run:

    make lint

Pull Request Checklist
----------------------

When you make a pull request, please copy this text into your first message 
of the pull request, which will create a checklist, which should be completed 
before the pull request is merged.

```
- [ ] Is it mergable?
- [ ] Did it pass the tests?
- [ ] If it introduces new functionality in scripts/ is it tested?
  Check for code coverage. To run code coverage on only the file you changed,
  for example `flotilla/compute/splicing.py`, use this command: 
  `py.test --cov flotilla/compute/splicing.py --cov-report term-missing flotilla/test/compute/test_splicing.py`
  which will show you which lines aren't covered by the tests.
- [ ] Do the new functions have descriptive 
  [numpydoc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
  style docstrings?
- [ ] If it adds a new plot, is it documented in the gallery?
- [ ] Is it well formatted? Look at `make pep8` and `make lint` output
- [ ] Is it documented in the doc/releases/?
- [ ] Was a spellchecker run on the source code and documentation after
  changes were made?
```