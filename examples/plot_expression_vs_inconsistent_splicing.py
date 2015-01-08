"""
Show percentage of splicing events whose psi scores are inconsistent between pooled and single
==============================================================================================
"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_expression_vs_inconsistent_splicing()