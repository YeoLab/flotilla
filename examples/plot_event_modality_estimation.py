"""
Plot the modality log-likelihoods and barplots during estimation
================================================================

See also
--------
:py:func:`Study.plot_event_modality_estimation`

"""
import flotilla
study = flotilla.embark('shalek2013')
study.plot_event_modality_estimation('chr8:97356415:97356600:-@chr8:97355689:97355825:-@chr8:97353054:97353130:-@chr8:97352177:97352339:-')