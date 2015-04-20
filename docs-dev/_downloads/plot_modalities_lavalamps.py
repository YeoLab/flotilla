"""
Plot bar graphs of percentage of splicing events in each modality
=================================================================

See also
--------
:py:func:`Study.plot_modalities_bars`

"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_modalities_lavalamps()