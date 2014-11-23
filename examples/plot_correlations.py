"""
Visualize global correlations between samples or features

See also
--------
:py:func:`Study.interactive_correlations`

"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_correlations()