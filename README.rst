flotilla
========

``flotilla`` is a Python package for visualizing transcriptome (RNA expression) data from hundreds of
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
  
  
Finally, ``flotilla`` is a platform for active collaboration between bioinformatics scientists and 
traditional "wet lab" scientists. Leveraging `interactive widgets <https://github.com/ipython/ipython/tree/master/examples/Interactive%20Widgets>`_ 
in the `IPython Notebook <http://ipython.org/notebook.html>`_, 
we have created tools for simple and streamlined data exploration including:

  * Subsetting sample groups and feature (genes/splicing events) groups
  * Dynamically adjusting parameters for analysis
  * Integrating external lists of features from the web or local files

These empower the "wet lab" scientists to ask questions on their own and gives bioniformatics
scientists a platform and share their analysis tools.


What flotilla is **not**
-----------------------

``flotilla`` is not a genomics pipeline. We expect that you have already generated
data tables for gene expression, isoform expression and metadata. ``flotilla`` only makes 
it easy to integrate all those data parts together once you have the pieces.
