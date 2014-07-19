"""
Visualize the result of a classifcation or regression algorithm on the data.
"""
import itertools

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .decomposition import PCAViz
from ..compute.predict import Classifier, Regressor
from .color import green


class PredictorBaseViz(object):

    _reducer_plotting_args = {}

    def set_reducer_plotting_args(self, rpa):
        self._reducer_plotting_args.update(rpa)

    def generate_scatter_table(self,
                               excel_out=None, external_xref=[]):
        """
        make a table to make scatterplots... maybe for plot.ly
        excelOut: full path to an excel output location for scatter data
        external_xref: list of tuples containing (attribute name, function to
        map row index -> an attribute)
        """
        raise NotImplementedError
        trait, classifier_name = self.data.trait, self.predictor.name
        X = self.X
        sorter = np.array([np.median(i[1]) - np.median(j[1]) for (i, j) in
                           itertools.izip(X[self.y[trait] == 0].iteritems(),
                                          X[self.y[trait] == 1].iteritems())])

        sort_by_sorter = X.columns[np.argsort(sorter)]
        c0_values = X[sort_by_sorter][self.y[trait] == 0]
        c1_values = X[sort_by_sorter][self.y[trait] == 1]

        x = []
        s = []
        y1 = []
        y2 = []
        field_names = ['x-position', 'probe intensity', "condition0",
                       "condition1"]
        n_default_fields = len(field_names)
        external_attrs = {}
        for external_attr_name, external_attr_fun in external_xref:
            external_attrs[external_attr_name] = []
            field_names.append(external_attr_name)

        for i, (a, b) in enumerate(
                itertools.izip(c0_values.iteritems(), c1_values.iteritems())):

            mn = np.mean(np.r_[a[1], b[1]])
            _ = [x.append(i) for _ in a[1]]
            _ = [s.append(mn) for val in a[1]]
            _ = [y1.append(val - mn) for val in a[1]]
            _ = [y2.append(np.nan) for val in a[1]]

            _ = [x.append(i) for _ in b[1]]
            _ = [s.append(mn) for val in b[1]]
            _ = [y1.append(np.nan) for val in b[1]]
            _ = [y2.append(val - mn) for val in b[1]]

            for external_attr_name, external_attr_fun in external_xref:
                external_attrs[external_attr_name].extend(
                    [external_attr_fun(i) for i in a[1].index])
                external_attrs[external_attr_name].extend(
                    [external_attr_fun(i) for i in b[1].index])

        zz = pd.DataFrame([x, s, y1, y2] + [external_attrs[attr] for attr in
                                            field_names[n_default_fields:]],
                          index=field_names)

        if excel_out is not None:
            try:
                E = pd.ExcelWriter('%s' % excel_out)
                zz.T.to_excel(E, self.descrip)
                E.save()
            except Exception as e:
                print "excel save failed with error %s" % e

        return zz

    def do_pca(self, ax=None, **plotting_kwargs):

        """plot kernel density of predictor scores and draw a vertical line
        where the cutoff was selected
        ax : matplotlib.axes.Axes
            ax to plot on. if None: plt.gca()
        """

        # assert trait in self.traits
        assert self.has_been_fit
        assert self.has_been_scored

        if ax is None:
            ax = plt.gca()

        local_plotting_kwargs = self._reducer_plotting_args
        local_plotting_kwargs.update(plotting_kwargs)
        pca = PCAViz(self.X.ix[:, self.important_features],
                     **local_plotting_kwargs)
        pca(ax=ax)
        return pca

    def __call__(self, score_coefficient=None, *args, **kwargs):

        if not self.has_been_fit:
            self.fit()
        self.score(score_coefficient)

        self.do_pca(*args, **kwargs)


class RegressorViz(Regressor, PredictorBaseViz):

    def check_a_feature(self, feature_name, trait, **violinplot_kwargs):
        """Make Violin Plots for a gene/probe's value in the sets defined in
        sets

        feature_name - gene/probe id. must be in the index of self._parent.X
        sets - list of sample ids
        violinplot_kwargs - extra parameters for violinplot

        returns a list of lists with values for feature_name in each set of
        sets
        """
        sns.violinplot(self.X[feature_name], linewidth=0, groupby=trait,
                       alpha=0.5, bw='silverman', inner='points',
                       names=None, **violinplot_kwargs)
        sns.despine()


class ClassifierViz(Classifier, PredictorBaseViz):
    """
    Visualize results from classification
    """

    def __call__(self, ax=None, feature_score_std_cutoff=None,
                 **plotting_kwargs):

        if not self.has_been_fit:
            self.fit()
            self.score()

        gs_x = 18
        gs_y = 12

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(18, 8))
            gs = GridSpec(gs_x, gs_y)
        else:
            gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())
            fig = plt.gcf()

        ax_pca = plt.subplot(gs[:, 2:])
        ax_scores = plt.subplot(gs[5:10, :2])

        ax_scores.set_xlabel("Feature Importance")
        ax_scores.set_ylabel("Density Estimate")
        self.plot_classifier_scores(self.data.trait, ax=ax_scores)
        pca = self.do_pca(ax=ax_pca, show_vectors=True,
                          **plotting_kwargs)
        plt.tight_layout()
        return pca

    def plot_classifier_scores(self,  ax=None, classifier_name=None):
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
                    % (self.data.trait_name, self.n_good_features_, self.oob_score_))
        ax.axvline(x=self.score_cutoff_, color=green)

        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
        sns.despine(ax=ax)
