"""
Perform classification on categorical traits
============================================

See also
--------
:py:func:`Study.interactive_classifier`

"""
import flotilla
study = flotilla.embark(flotilla._brainspan)
study.plot_classifier('structure_name: cerebellar cortex')