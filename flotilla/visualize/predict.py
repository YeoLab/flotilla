"""
Visualize the result of a classifcation or regression algorithm on the data.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import seaborn as sns

from ..compute.predict import Classifier, Regressor, PredictorBase
from .color import green
from .decomposition import DecompositionViz


class PredictorBaseViz(PredictorBase):
    _reducer_plotting_args = {}

    def plot(self, **pca_plotting_kwargs):
        if not self.has_been_fit:
            self.fit()

        gs_x = 18
        gs_y = 12

        ax = None if 'ax' not in pca_plotting_kwargs \
            else pca_plotting_kwargs['ax']

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(18, 8))
            gs = GridSpec(gs_x, gs_y)
        else:
            gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())

        ax_scores = plt.subplot(gs[5:10, :2])
        ax_scores.set_xlabel("Feature Importance")
        ax_scores.set_ylabel("Density Estimate")

        if 'show_vectors' not in pca_plotting_kwargs:
            pca_plotting_kwargs['show_vectors'] = True

        ax_pca = plt.subplot(gs[:, 2:])
        pca_plotting_kwargs['ax'] = ax_pca

        self.plot_scores(ax=ax_scores)
        pcaviz = self.do_pca(**pca_plotting_kwargs)
        plt.tight_layout()

        return pcaviz

    def set_reducer_plotting_args(self, rpa):
        self._reducer_plotting_args.update(rpa)

    def do_pca(self, **plotting_kwargs):
        # assert trait in self.traits
        assert self.has_been_fit
        assert self.has_been_scored

        ax = plotting_kwargs.pop('ax', plt.gca())
        local_plotting_kwargs = self._reducer_plotting_args
        local_plotting_kwargs.update(plotting_kwargs)
        pca = self.pca()

        if self.categorical:
            groupby = self.dataset.trait.align(self.y, join='right')[0]
        else:
            groupby = self.y

        pcaviz = DecompositionViz(pca.reduced_space, pca.components_,
                                  pca.explained_variance_ratio_,
                                  feature_renamer=self.feature_renamer,
                                  groupby=groupby,
                                  singles=self.singles,
                                  pooled=self.pooled,
                                  outliers=self.outliers)
        pcaviz.plot(ax=ax, **local_plotting_kwargs)
        return pcaviz

    def plot_scores(self, ax=None):

        """
        plot kernel density of predictor scores and draw a vertical line where
        the cutoff was selected
        ax - ax to plot on. if None: plt.gca()
        """

        if ax is None:
            ax = plt.gca()

        # for trait in traits:
        sns.kdeplot(self.scores_, shade=True, ax=ax,
                    label="%s\n%d features\noob:%.2f"
                          % (self.dataset.trait_name, self.n_good_features_,
                             self.oob_score_))
        ax.axvline(x=self.score_cutoff_, color=green)

        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
        sns.despine(ax=ax)


class RegressorViz(Regressor, PredictorBaseViz):
    pass


class ClassifierViz(Classifier, PredictorBaseViz):
    """
    Visualize results from classification
    """

    def check_a_feature(self, feature_name, **violinplot_kwargs):
        """Make Violin Plots for a gene/probe's value in the sets defined in
        sets

        feature_name - gene/probe id. must be in the index of self._parent.X
        sets - list of sample ids
        violinplot_kwargs - extra parameters for violinplot

        returns a list of lists with values for feature_name in each set of
        sets
        """
        sns.violinplot(self.X[feature_name], linewidth=0, groupby=self.y,
                       alpha=0.5, bw='silverman', inner='points',
                       names=None, **violinplot_kwargs)
        sns.despine()
