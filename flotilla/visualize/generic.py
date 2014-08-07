import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def violinplot(data, groupby=None, color=None, ax=None, pooled_data=None,
               order=None, violinplot_kws=None, title=None,
               label_pooled=True, outliers=None, data_type=None):
    """
    Parameters
    ----------
    data : pandas.Series
        The main data to plot
    groupby : dict-like
        How to group the samples (e.g. by phenotype)
    color : list

    Returns
    -------


    Raises
    ------
    """
    data_type = 'none' if data_type is None else data_type
    splicing = 'splicing'.startswith(data_type)

    violinplot_kws = {} if violinplot_kws is None else violinplot_kws
    violinplot_kws.setdefault('alpha', 0.75)

    if ax is None:
        ax = plt.gca()

    _violinplot_data(data, groupby=groupby, color=color, ax=ax, order=order,
                     violinplot_kws=violinplot_kws, splicing=splicing)
    if pooled_data is not None and groupby is not None:
        grouped = pooled_data.groupby(groupby)
        if order is not None:
            for i, name in enumerate(order):
                try:
                    subset = pooled_data.ix[grouped.groups[name]]
                    plot_pooled_dot(ax, subset, x_offset=i, label=label_pooled)
                except KeyError:
                    pass
        else:
            plot_pooled_dot(ax, pooled_data)

    if outliers is not None:
        outlier_violinplot_kws = violinplot_kws

        # make sure this is behind the non outlier data
        outlier_violinplot_kws['zorder'] = -1
        _violinplot_data(outliers, groupby=groupby, color='lightgrey', ax=ax,
                         order=order, violinplot_kws=outlier_violinplot_kws,
                         splicing=splicing)

    if splicing:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('$\Psi$')

    if title is not None:
        ax.set_title(title)

    if order is not None:
        ax.set_xlim(-.5, len(order) - .5)

    if groupby is not None and order is not None:
        sizes = data.groupby(groupby).size()
        xticks = range(len(order))
        xticklabels = ['{}\nn={}'.format(group, sizes[group])
                       if group in sizes else '{}\nn=0'.format(group)
                       for group in order]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    sns.despine()


def _violinplot_data(data, groupby=None, order=None, violinplot_kws=None,
                     color=None, ax=None, splicing=False):
    """Plot a single groups violinplot.

    Separated out so real data plotting and outlier plotting works the same
    """
    data = data.dropna()

    single_points = data.groupby(groupby).filter(lambda x: len(x) < 2)
    data = data.groupby(groupby).filter(lambda x: len(x) > 1)

    # Check that all the groups are represented, if not, add some data out of
    # range to the missing group
    verified_color = color
    if groupby is not None and order is not None:
        verified_groups = data.groupby(groupby).size().keys()
        verified_order = [x for x in order if x in verified_groups]

        positions = [i for i, x in enumerate(order) if x in verified_groups]

        single_groups = single_points.groupby(groupby).size().keys()
        single_positions = dict((x, i) for i, x in enumerate(order) if
                                x in single_groups)

        if not mpl.colors.is_color_like(color):
            verified_color = [x for i, x in enumerate(color)
                              if order[i] in verified_groups]
            single_color = dict((group, c) for i, (c, group) in
                                enumerate(zip(color, single_groups))
                                if group in single_groups)
        else:
            single_color = dict((group, color) for group in single_groups if
                                group in single_groups)
    else:
        verified_order = order
        positions = None

        single_positions = None
        single_color = None

    violinplot_kws = {} if violinplot_kws is None else violinplot_kws

    # Add a tiny amount of random noise in case the values are all identical,
    # Otherwise we get a LinAlg error.
    data += np.random.uniform(0, 0.001, data.shape[0])

    inner = 'points' if splicing else 'box'
    sns.violinplot(data, groupby=groupby, bw=0.1, inner=inner,
                   color=verified_color, linewidth=0.5, order=verified_order,
                   ax=ax, positions=positions, **violinplot_kws)

    if single_points is not None:
        for group, y in single_points.groupby(groupby):
            x = single_positions[group]
            c = single_color[group]
            ax.scatter([x], [y], color=c, s=50)
            ax.annotate(y.index[0], (x, y), textcoords='offset points',
                        xytext=(7, 0), fontsize=14)


def plot_pooled_dot(ax, pooled, x_offset=0, label=True):
    """
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    pooled : pandas.Series
        Pooled data of this gene

    Returns
    -------


    Raises
    ------
    """
    pooled = pooled.dropna()
    try:
        xs = np.zeros(pooled.shape[0])
    except AttributeError:
        xs = np.zeros(1)
    xs += x_offset
    ax.plot(xs, pooled, 'o', color='#262626')

    if label:
        for x, y in zip(xs, pooled):
            if np.isnan(y):
                continue
            ax.annotate('pooled', (x, y), textcoords='offset points',
                        xytext=(7, 0), fontsize=14)