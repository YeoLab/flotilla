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


class _ModalityEstimatorPlotter(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(5 * 2, 3 * 2))
        self.ax_violin = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=1)
        self.ax_loglik = plt.subplot2grid((3, 5), (0, 1), rowspan=3, colspan=3)
        self.ax_bayesfactor = plt.subplot2grid((3, 5), (0, 4), rowspan=3,
                                               colspan=1)

    def plot(self, event, logliks, logsumexps, modality_colors,
             renamed=''):
        modality = logsumexps.idxmax()

        sns.violinplot(event.dropna(), bw=0.2, ax=self.ax_violin,
                       color=modality_colors[modality])

        self.ax_violin.set_ylim(0, 1)
        self.ax_violin.set_title('Guess: {}'.format(modality))
        self.ax_violin.set_xticks([])
        self.ax_violin.set_yticks([0, 0.5, 1])
        # self.ax_violin.set_xlabel(renamed)

        for name, loglik in logliks.iteritems():
            # print name,
            self.ax_loglik.plot(loglik, 'o-', label=name,
                                color=modality_colors[name])
            self.ax_loglik.legend(loc='best')
        self.ax_loglik.set_title('Log likelihoods at different '
                                 'parameterizations')
        self.ax_loglik.grid()
        self.ax_loglik.set_xlabel('phantom', color='white')

        for i, (name, height) in enumerate(logsumexps.iteritems()):
            self.ax_bayesfactor.bar(i, height, label=name,
                                    color=modality_colors[name])
        self.ax_bayesfactor.set_title('$\log$ Bayes factors')
        self.ax_bayesfactor.set_xticks([])
        self.ax_bayesfactor.grid()
        self.fig.tight_layout()
        self.fig.text(0.5, .025, '{} ({})'.format(event.name, renamed),
                      fontsize=10, ha='center', va='bottom')
        sns.despine()
        return self


class ModalitiesViz(object):
    """Visualize results of modality assignments"""
    modality_colors = {'bimodal': seaborn_colors[3],
                       'Psi~0': seaborn_colors[0],
                       'Psi~1': seaborn_colors[2],
                       'middle': seaborn_colors[1],
                       'ambiguous': seaborn_colors[4]}

    modality_order = ['Psi~0', 'middle', 'Psi~1', 'bimodal', 'ambiguous']

    colors = [modality_colors[modality] for modality in
              modality_order]

    def plot_reduced_space(self, binned_reduced, modality_assignments,
                           ax=None, title=None, xlabel='', ylabel=''):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # For easy aliasing
        X = binned_reduced

        for modality, df in X.groupby(modality_assignments, axis=0):
            color = self.modality_colors[modality]
            ax.plot(df.ix[:, 0], df.ix[:, 1], 'o', color=color, alpha=0.7,
                    label=modality)

        sns.despine()
        xmax, ymax = X.max()
        ax.set_xlim(0, 1.05 * xmax)
        ax.set_ylim(0, 1.05 * ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        if title is not None:
            ax.set_title(title)

    def bar(self, counts, phenotype_to_color=None, ax=None, percentages=True):
        """Draw barplots grouped by modality of modality percentage per group

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        if percentages:
            counts = 100 * (counts.T / counts.T.sum()).T

        # with sns.set(style='whitegrid'):
        if ax is None:
            ax = plt.gca()

        full_width = 0.8
        width = full_width / counts.shape[0]
        for i, (group, series) in enumerate(counts.iterrows()):
            left = np.arange(len(self.modality_order)) + i * width
            height = [series[i] if i in series else 0
                      for i in self.modality_order]
            color = phenotype_to_color[group]
            ax.bar(left, height, width=width, color=color, label=group,
                   linewidth=.5, edgecolor='k')
        ylabel = 'Percentage of events' if percentages else 'Number of events'
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(self.modality_order)) + full_width / 2)
        ax.set_xticklabels(self.modality_order)
        ax.set_xlabel('Splicing modality')
        ax.set_xlim(0, len(self.modality_order))
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        sns.despine()

    def event_estimation(self, event, logliks, logsumexps, renamed=''):
        """Show the values underlying bayesian modality estimations of an event

        Parameters
        ----------


        Returns
        -------


        Raises
        ------
        """
        plotter = _ModalityEstimatorPlotter()
        plotter.plot(event, logliks, logsumexps, self.modality_colors,
                     renamed=renamed)
        return plotter


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
    plot_kws.setdefault('rasterized', True)

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


def nmf_space_transitions(nmf_space_positions, feature_id,
                          phenotype_to_color, phenotype_to_marker, order,
                          ax=None, xlabel=None, ylabel=None):
    """Plot 2d space traveled by individual splicing events

    Parameters
    ----------
    nmf_space_positions : pandas.DataFrame
        A dataframe with a multiindex of (event, phenotype) and columns of
        x- and y- position, respectively
    feature_id : str
        Unique identifier of the feature to plot
    phenotype_to_color : dict
        Mapping of the phenotype name to a color
    phenotype_to_marker : dict
        Mapping of the phenotype name to a plotting symbol
    order : tuple
        Order in which to plot the phenotypes (e.g. if there is a biological
        ordering)
    ax : matplotlib.Axes object, optional
        An axes to plot these onto. If not provided, grabs current axes
    xlabel : str, optional
        How to label the x-axis
    ylabel : str, optional
        How to label the y-axis
    """
    df = nmf_space_positions.ix[feature_id]

    if ax is None:
        ax = plt.gcf()

    for color, s in df.groupby(phenotype_to_color, axis=0):
        phenotype = s.index[0]
        marker = phenotype_to_marker[phenotype]
        ax.plot(s.pc_1, s.pc_2, color=color, marker=marker, markersize=14,
                alpha=0.75, label=phenotype, linestyle='none')

    # ax.scatter(df.ix[:, 0], df.ix[:, 1], color=color, s=100, alpha=0.75)
    # ax.legend(points, df.index.tolist())
    ax.set_xlim(0, nmf_space_positions.ix[:, 0].max() * 1.05)
    ax.set_ylim(0, nmf_space_positions.ix[:, 1].max() * 1.05)

    x = [df.ix[pheno, 0] for pheno in order if pheno in df.index]
    y = [df.ix[pheno, 1] for pheno in order if pheno in df.index]

    ax.plot(x, y, zorder=-1, color='#262626', alpha=0.5, linewidth=1)
    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        ax.set_yticks([])
