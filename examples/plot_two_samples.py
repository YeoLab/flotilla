"""
Compare gene expression in two samples
======================================
"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_two_samples('S1', 'S2')