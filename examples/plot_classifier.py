"""
Perform classification on categorical traits
============================================

See also
--------
:py:func:`Study.interactive_classifier`

"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_classifier('maturity: immature')