"""
Visualize principal component analysis dimensionality reduction
================================================================

See also
--------
:py:func:`Study.interactive_pca`

"""
import flotilla
study = flotilla.embark(flotilla._brainspan)
study.plot_pca(plot_violins=False)