
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

    pip install -e

Git branching
-------------

We use the [git-flow](http://nvie.com/posts/a-successful-git-branching-model/) model.

We use the [`gitflow`](https://github.com/nvie/gitflow) model of branching
and features. The "production release" repo is `master` and the '"next release"
development' repo is `dev`. Everything else is default:

```
Which branch should be used for bringing forth production releases?
   - master
Branch name for production releases: [master]
Branch name for "next release" development: [develop] dev

How to name your supporting branch prefixes?
Feature branches? [feature/]
Release branches? [release/]
Hotfix branches? [hotfix/]
Support branches? [support/]
Version tag prefix? []
```


So if you have a feature
(called `myfeature` as an example) you want to add, please add it off of the
`dev` branch, as so:

    git flow feature start myfeature

This creates the branch `feature/myfeature` as a copy off of `dev`.

... Make changes to files, commit them ...

When you're done working on your feature, do

    git flow feature finish myfeature

Which will merge the branch `feature/myfeature` with `dev`,
and remove the branch `feature/myfeature` in one command!


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
check out the instructions [here](https://github
.com/YeoLab/PyCharm-Python-Templates) for how to install and use them.

Testing
=======

In the source directory (wherever you cloned `flotilla` to that has this README.md file), do:

    make test

This will run the unit test suite.

Coverage
--------

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



What flotilla is not
====================

Flotilla does not claim to solve the data management problem of biology,
i.e. how you store all the data associated with a particular study that was
investigating a specific biological question. Flotilla only makes it easy to
integrate all those data parts together.
