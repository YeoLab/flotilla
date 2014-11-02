[![Build Status](https://travis-ci.org/YeoLab/flotilla.svg?branch=master)](https://travis-ci.org/YeoLab/flotilla)[![Coverage Status](https://img.shields.io/coveralls/YeoLab/flotilla.svg)](https://coveralls.io/r/YeoLab/flotilla?branch=master)[![License](https://pypip.in/license/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![Downloads](https://pypip.in/download/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![Latest Version](https://pypip.in/version/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![DOI](https://zenodo.org/badge/6604/YeoLab/flotilla.png)](http://dx.doi.org/10.5281/zenodo.12230)

flotilla
========

![flotilla Logo](https://github.com/YeoLab/flotilla/blob/master/flotilla.png)


[Installation instructions.](INSTALL.md)


Test interactive features with example data:
------------

We have prepared a slice of the full dataset for testing and demonstration purposes.

Run each of the following code lines in its own ipython notebook cell for an interactive feature.

    import flotilla
    test_study = flotilla.embark('http://sauron.ucsd.edu/flotilla_projects/neural_diff_chr22/datapackage.json')

    test_study.interactive_pca()

    test_study.interactive_graph()

    test_study.interactive_classifier()

    test_study.interactive_lavalamp_pooled_inconsistent()

IMPORTANT NOTE: for this test,several failures are expected since the test set is small.
Adjust parameters to explore valid parameter spaces.
For example, you can manually select `all_genes` as the `feature_subset`
from the drop-down menu that appears after running these interactive functions.


![NumFOCUS logo](http://numfocus.org/theme/img/numfocus_logo.png)

Proudly sponsored by a NumFOCUS John Hunter Technical Fellowship to Olga
Botvinnik.
