"""
Visualize principal component analysis dimensionality reduction
================================================================

"""
import flotilla
study = flotilla.embark(flotilla._shalek2013)
study.plot_classifier(plot_violins=False)