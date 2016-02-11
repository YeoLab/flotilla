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


def lavalamp(psi, yticks=(0, 0.5, 1), x_offset=0, title='', ax=None,
             switchy_score_psi=None, marker='o', markersize=10,
             markeredgewidth=0.1, markeredgecolor='#262626',
             rasterized=True, alpha=0.2, **kwargs):
    """Make a 'lavalamp' scatter plot of many splicing events

    Useful for visualizing many splicing events at once.

    Parameters
    ----------
    psi : array
        A (n_samples, n_events) matrix either as a numpy array or as a pandas
        DataFrame
    yticks : array
        Which yticks to display
    x_offset : numeric
        How much to offset the x-values off of 0 (fractions are best).
        Useful if you are plotting several celltypes with separate calls to
        lavalamp()
    title : str
        Title of the plot. Default ''
    ax : matplotlib.Axes object
        The axes to plot on. If not provided, will be created
    switchy_score_psi : pandas.DataFrame
        The psi scores to sort on for the plotting order. By default use the
        psi provided, but sometimes you want to plot multiple psi scores on
        the same plot, with the same events.
    marker : str
        A valid matplotlib plotting symbol marker. Default is 'o' (circle)
    markersize : int
        How big the plotting symbol should be
    markeredgewidth : float
        Linewidth of the marker outline
    markeredgecolor : matplotlib color
        Color of the marker outline
    rasterized : bool
        If True, save the scatterplot as a pixel-based (rather than
        vector-based) plot to save space. True by default.
    alpha : float
        How transparent to plot the markers (1 is opaque)
    kwargs : dict
        Keyword arguments to supply to plot()

    Returns
    -------
    fig : matplotlib.Figure
        A figure object for saving.
    """
    if psi.shape[1] == 0:
        return

    if ax is None:
        ax = plt.gca()

    y = as_numpy(psi.dropna(how='all', axis=1))

    if switchy_score_psi is not None:
        switchy_score_y = as_numpy(switchy_score_psi)
    else:
        switchy_score_y = y

    order = get_switchy_score_order(switchy_score_y)
    y = y[:, order]

    n_samples, n_events = y.shape
    x = np.vstack((np.arange(n_events) for _ in xrange(n_samples)))
    # .astype(float) is to get rid of a deprecation warning
    x = x.astype(float)
    x += x_offset

    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1

    ax.plot(x, y, alpha=alpha, rasterized=rasterized, markersize=markersize,
            marker=marker, linestyle='None',
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            **kwargs)
    sns.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(n_events))
    ax.set_xticks([])

    ax.set_xlim(-0.5, xmax + .5)
    ax.set_ylim(0, 1)
    ax.set_yticks(yticks)
    ax.set_title(title)
    return ax


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
    pooled_kwargs = {'alpha': 0.5, 'markeredgecolor': 'k',
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
                 ax=ax_inconsistent, kwargs=pooled_kwargs)
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
             ax=ax_consistent, kwargs=pooled_kwargs)
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
        ax = plt.gca()

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
