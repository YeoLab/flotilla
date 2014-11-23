"""
Visualize hierarchical relationships between samples and features

See also
--------
:py:func:`Study.interactive_graph`

"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_graph(cov_cut=0.5)