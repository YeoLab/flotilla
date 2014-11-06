[![Build Status](https://travis-ci.org/YeoLab/flotilla.svg?branch=master)](https://travis-ci.org/YeoLab/flotilla)[![Coverage Status](https://img.shields.io/coveralls/YeoLab/flotilla.svg)](https://coveralls.io/r/YeoLab/flotilla?branch=master)[![License](https://pypip.in/license/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![Downloads](https://pypip.in/download/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![Latest Version](https://pypip.in/version/flotilla/badge.svg)](https://pypi.python.org/pypi/flotilla/)[![DOI](https://zenodo.org/badge/6604/YeoLab/flotilla.png)](http://dx.doi.org/10.5281/zenodo.12230)

flotilla
========

![flotilla Logo](flotilla.png)

Getting flotilla
================

Instructions for the best way to obtain `flotilla` are [here](INSTALL.md).

What is flotilla?
=================

`flotilla` is a Python package for visualizing transcriptome (RNA expression) data from hundreds of
samples. We include utilities to perform common tasks on these large data matrices, including:
 
  * Dimensionality reduction
  * Classification and Regression
  * Outlier detection
  * Network graphs from covariance
  * Hierarchical clustering
  
And common tasks for biological data including:

  * Renaming database features to gene symbols
  * Coloring/marking samples based on experimental phenotype
  * Removing poor-quality samples (technical outliers)
  
  
Finally, `flotilla` is a platform for active collaboration between bioinformatics scientists and 
traditional "wet lab" scientists. Leveraging [interactive widgets](https://github.com/ipython/ipython/tree/master/examples/Interactive%20Widgets) 
in the [iPython Notebook](http://ipython.org/notebook.html), 
we have created tools for simple and streamlined data exploration including:

  * Subsetting sample groups and feature (genes/splicing events) groups
  * Dynamically adjusting parameters for analysis
  * Integrating external lists of features from the web or local files

These empower the "wet lab" scientists to ask questions on their own and gives bioniformatics
scientists a platform and share their analysis tools.


What flotilla is **not**
========================

`flotilla` is not a genomics pipeline. We expect that you have already generated
data tables for gene expression, isoform expression and metadata. `flotilla` only makes 
it easy to integrate all those data parts together once you have the pieces.

Learn how to use flotilla
=========================
Please refer to our [talks](talks/) or [tutorials](tutorials/) to learn more about how you can
apply our tools to your data.

Quick Start:
============

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


Problems? Questions?
====================

We invite your input! Please leave any feedback on our [issues page](https://github.com/YeoLab/flotilla/issues).

![NumFOCUS logo](http://numfocus.org/theme/img/numfocus_logo.png)

Proudly sponsored by a NumFOCUS John Hunter Technical Fellowship to Olga
Botvinnik.
