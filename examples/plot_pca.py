"""
Visualize principal component analysis dimensionality reduction
================================================================

"""
import flotilla
study = flotilla.embark(flotilla._neural_diff_chr22)
study.plot_pca(plot_violins=False)