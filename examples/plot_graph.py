"""
Visualize hierarchical relationships between samples and features

See also
--------
:py:func:`Study.interactive_graph`

"""
import flotilla
study = flotilla.embark(flotilla._brainspan)
study.plot_graph(cov_std_cut=3, degree_cut=5)