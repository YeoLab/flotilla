"""
Splicing-specific visualization classes and methods
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from .color import red, blue, purple, grey, green
from ..compute.splicing import get_switchy_score_order
from ..util import as_numpy

seaborn_colors = map(mpl.colors.rgb2hex, sns.color_palette('deep'))


class ModalitiesViz(object):
    """Visualize results of modality assignments
    """
    modalities_colors = {'bimodal': seaborn_colors[3],
                         'excluded': seaborn_colors[0],
                         'included': seaborn_colors[2],
                         'middle': seaborn_colors[1],
                         'ambiguous': 'lightgrey',
                         'uniform': seaborn_colors[4]}

    modalities_order = ['excluded', 'uniform', 'bimodal', 'middle',
                        'included', 'ambiguous']

    colors = [modalities_colors[modality] for modality in
              modalities_order]

    def plot_reduced_space(self, binned_reduced, modalities_assignments,
                           ax=None, title=None, xlabel='', ylabel=''):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # For easy aliasing
        X = binned_reduced

        # import pdb
        # pdb.set_trace()

        for modality, df in X.groupby(modalities_assignments, axis=0):
            color = self.modalities_colors[modality]
            ax.plot(df.ix[:, 0], df.ix[:, 1], 'o', color=color, alpha=0.25,
                    label=modality)

        sns.despine()
        xmax, ymax = X.max()
        ax.set_xlim(0, 1.05 * xmax)
        ax.set_ylim(0, 1.05 * ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        if title is not None:
            ax.set_title(title)

    def bar(self, modalities_counts, ax=None, i=0, normed=True, legend=True):
        """Draw the barplot of a single modalities_count

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        modalities_counts = modalities_counts[self.modalities_order]

        if normed:
            modalities_counts = \
                modalities_counts / modalities_counts.sum().astype(float)

        lefts = np.ones(modalities_counts.shape) * i

        heights = modalities_counts
        bottoms = np.zeros(modalities_counts.shape)
        bottoms[1:] = modalities_counts.cumsum()[:-1]
        labels = self.modalities_order

        for left, height, bottom, color, label in zip(lefts, heights, bottoms,
                                                      self.colors, labels):
            ax.bar(left, height, bottom=bottom, color=color, label=label,
                   alpha=0.75)

        if legend:
            ax.legend()
        sns.despine()

    def event(self, feature_id, sample_groupby, group_colors, group_order,
              ax=None):
        """Plot a single splicing event's changes in DataFrameNMF space, and its
        violin plots

        """
        pass


def lavalamp(psi, color=None, x_offset=0, title='', ax=None,
             switchy_score_psi=None, marker='o', plot_kws=None,
             yticks=None):
    """Make a 'lavalamp' scatter plot of many splicing events

    Useful for visualizing many splicing events at once.

    Parameters
    ----------
    psi : array
        A (n_events, n_samples) matrix either as a numpy array or as a pandas
        DataFrame
    color : matplotlib color
        Color of the scatterplot. Defaults to a dark teal
    x_offset : numeric or None
        How much to offset the x-values off of 1. Useful for plotting several
        celltypes at once.
    title : str
        Title of the plot. Default ''
    ax : matplotlib.Axes object
        The axes to plot on. If not provided, will be created
    switchy_score_psi : pandas.DataFrame
        The psi scores to sort on for the plotting order. By default use the
        psi provided, but sometimes you want to plot multiple psi scores on
        the same plot, with the same events.
    marker : str
        A valid matplotlib marker. Default is 'd' (thin diamond)
    plot_kws : dict
        Keyword arguments to supply to plot()

    Returns
    -------
    fig : matplotlib.Figure
        A figure object for saving.
    """
    if psi.shape[1] == 0:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))

    color = seaborn_colors[0] if color is None else color
    plot_kws = {} if plot_kws is None else plot_kws
    plot_kws.setdefault('color', color)
    plot_kws.setdefault('alpha', 0.2)
    plot_kws.setdefault('markersize', 10)
    plot_kws.setdefault('marker', marker)
    plot_kws.setdefault('linestyle', 'None')
    plot_kws.setdefault('markeredgecolor', '#262626')
    plot_kws.setdefault('markeredgewidth', .1)

    y = as_numpy(psi.dropna(how='all', axis=1))

    if switchy_score_psi is not None:
        switchy_score_y = as_numpy(switchy_score_psi)
    else:
        switchy_score_y = y

    order = get_switchy_score_order(switchy_score_y)
    y = y[:, order]

    n_samples, n_events = y.shape
    # .astype(float) is to get rid of a deprecation warning
    x = np.vstack((np.arange(n_events) for _ in xrange(n_samples)))
    x = x.astype(float)
    x += x_offset

    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1

    ax.plot(x, y, **plot_kws)
    sns.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(n_events))
    ax.set_xticks([])

    ax.set_xlim(-0.5, xmax + .5)
    ax.set_ylim(0, 1)
    if yticks is None:
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_yticks(yticks)
    ax.set_title(title)


def hist_single_vs_pooled_diff(diff_from_singles, diff_from_singles_scaled,
                               color=None, title='', nbins=50, hist_kws=None):
    """Plot a histogram of both the original difference difference of psi
    scores from the pooled to the singles, and the scaled difference

    """
    hist_kws = {} if hist_kws is None else hist_kws

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    dfs = (diff_from_singles, diff_from_singles_scaled)
    names = ('total_diff', 'scaled_diff')

    for ax, df, name in zip(axes, dfs, names):
        vmin = df.min().min()
        vmax = df.max().max()
        ax.hist(df.values.flat, bins=np.linspace(vmin, vmax, nbins),
                color=color, edgecolor='white', linewidth=0.5, **hist_kws)
        ax.set_title(title)
        # ax.set_title('{}, {}'.format(celltype, name))
        ax.grid(which='y', color='white')
    sns.despine()


def lavalamp_pooled_inconsistent(singles, pooled, pooled_inconsistent,
                                 color=None, percent=None):
    fig, axes = plt.subplots(nrows=2, figsize=(16, 8))
    ax_inconsistent = axes[0]
    ax_consistent = axes[1]
    plot_order = \
        pooled_inconsistent.sum() / pooled_inconsistent.count().astype(float)
    plot_order.sort()

    color = seaborn_colors[0] if color is None else color
    pooled_plot_kws = {'alpha': 0.5, 'markeredgecolor': 'k',
                       'markerfacecolor': 'none', 'markeredgewidth': 1}

    pooled = pooled.dropna(axis=1, how='all')

    suffix = ' of events measured in both pooled and single'

    ax_inconsistent.set_xticks([])
    ax_consistent.set_xticks([])

    try:
        singles_values = singles.ix[:, pooled_inconsistent.columns].values
        lavalamp(singles_values, color=color, ax=ax_inconsistent)
        lavalamp(pooled.ix[:, pooled_inconsistent.columns], marker='o',
                 color='k',
                 switchy_score_psi=singles_values,
                 ax=ax_inconsistent, plot_kws=pooled_plot_kws)
        title_suffix = '' if percent is None else ' ({:.1f}%){}'.format(
            percent, suffix)
        ax_inconsistent.set_title('Pooled splicing events inconsistent '
                                  'with singles{}'.format(title_suffix))
    except IndexError:
        # There are no inconsistent events
        pass

    singles = singles.dropna(axis=1, how='all')
    consistent_events = singles.columns[
        ~singles.columns.isin(pooled_inconsistent.columns)]
    lavalamp(singles.ix[:, consistent_events], color=color, ax=ax_consistent)
    lavalamp(pooled.ix[:, consistent_events], color='k', marker='o',
             switchy_score_psi=singles.ix[:, consistent_events],
             ax=ax_consistent, plot_kws=pooled_plot_kws)
    title_suffix = '' if percent is None else ' ({:.1f}%){}'.format(
        100 - percent, suffix)
    ax_consistent.set_title('Pooled splicing events consistent with singles{}'
                            .format(title_suffix))
    sns.despine()

