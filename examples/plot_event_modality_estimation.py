"""
Plot the modality log-likelihoods and barplots during estimation
================================================================

See also
--------
:py:func:`Study.plot_event_modality_estimation`

"""
import flotilla
study = flotilla.embark('shalek2013')
study.plot_event_modality_estimation('chr11:106616096:106616365:+@chr11:106622497:106622610:+@chr11:106625170:106625262:+')