"""
Plot splicing of a gene in all phenotypes
=========================================

In each column of the phenotype, the pooled samples are plotted as black dots
and the outliers are plotted as grey shadows.

``plot_event`` will plot ALL splicing events which are associated with that
feature ID, as in this case, there are multiple splicing events on the gene
RBFOX2.

"""
import flotilla
study = flotilla.embark(flotilla._neural_diff_chr22)
study.plot_event('RBFOX2')