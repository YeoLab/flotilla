"""
Compare gene expression in two samples
======================================
"""
import flotilla
study = flotilla.embark(flotilla._neural_diff_chr22)
study.plot_two_features('EWSR1', 'RBFOX2')