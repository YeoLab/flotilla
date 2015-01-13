"""
Plot splicing events on NMF space, colored by their modality in each celltype
=============================================================================

See also
--------
:py:func:`Study.plot_modalities_reduced`

"""
import flotilla
study = flotilla.embark('shalek2013')
study.plot_modalities_reduced()