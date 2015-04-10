"""
Compare gene expression in two features
======================================
"""
import flotilla
study = flotilla.embark(flotilla._brainspan)
study.plot_two_features('FOXP1', 'FOXJ1')