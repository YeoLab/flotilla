import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def violinplot(data, groupby=None, color_ordered=None, ax=None,
               pooled_data=None,
               order=None, violinplot_kws=None, title=None,
               label_pooled=False, outliers=None, data_type=None):
    """
    Parameters
    ----------
    data : pandas.Series
        The main data to plot
    groupby : dict-like
        How to group the samples (e.g. by phenotype)
    color_ordered : list

    Returns
    -------


    Raises
    ------
    """
    # import pdb; pdb.set_trace()
    data_type = 'none' if data_type is None else data_type
    splicing = 'splicing'.startswith(data_type)

    violinplot_kws = {} if violinplot_kws is None else violinplot_kws
    violinplot_kws.setdefault('alpha', 0.75)

    if ax is None:
        ax = plt.gca()

    _violinplot_single_dataset(data, groupby=groupby, color=color_ordered,
                               ax=ax, order=order,
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
        _violinplot_single_dataset(outliers, groupby=groupby, color='lightgrey',
                                   ax=ax,
                                   order=order,
                                   violinplot_kws=outlier_violinplot_kws,
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
        sizes = data.dropna().groupby(groupby).size()
        xticks = range(len(order))
        xticklabels = ['{}\nn={}'.format(group, sizes[group])
                       if group in sizes else '{}\nn=0'.format(group)
                       for group in order]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    sns.despine()


def _violinplot_single_dataset(data, groupby=None, order=None,
                               violinplot_kws=None, color=None, ax=None,
                               splicing=False):
    """Plot a single set of violinplot.

    Separated out so real data plotting and outlier plotting works the same
    """
    data = data.dropna()
    if data.empty:
        return

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
    if len(data) > 0:
        sns.violinplot(data, groupby=groupby, bw=0.1, inner=inner,
                       color=verified_color, linewidth=0.5,
                       order=verified_order,
                       ax=ax, positions=positions, **violinplot_kws)

    if single_points is not None:
        for group, y in single_points.groupby(groupby):
            x = single_positions[group]
            c = single_color[group]
            ax.scatter([x], [y], color=c, s=50)
            ax.annotate(y.index[0], (x, y), textcoords='offset points',
                        xytext=(7, 0), fontsize=14)


def plot_pooled_dot(ax, pooled, x_offset=0, label=False):
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


def nmf_space_transitions(nmf_space_positions, feature_id,
                          phenotype_to_color, phenotype_to_marker, order,
                          ax=None, xlabel=None, ylabel=None):
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

    x = [df.ix[phenotype, 0] for phenotype in order if phenotype in df.index]
    y = [df.ix[phenotype, 1] for phenotype in order if phenotype in df.index]

    ax.plot(x, y, zorder=-1, color='#262626', alpha=0.5, linewidth=1)
    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        ax.set_yticks([])


def simple_twoway_scatter(sample1, sample2, **kwargs):
    """Plot a two-dimensional scatterplot between two samples

    Parameters
    ----------
    sample1 : pandas.Series
        Data to plot on the x-axis
    sample2 : pandas.Series
        Data to plot on the y-axis
    Any other keyword arguments valid for seaborn.jointplot

    Returns
    -------
    jointgrid : seaborn.axisgrid.JointGrid
        Returns a JointGrid instance

    See Also
    -------
    seaborn.jointplot

    """
    joint_kws = kwargs.pop('joint_kws', {})

    kind = kwargs.pop('kind', 'scatter')
    marginal_kws = kwargs.pop('marginal_kws', {})
    if kind == 'scatter':
        vmin = min(sample1.min(), sample2.min())
        vmax = max(sample1.max(), sample2.max())
        bins = np.linspace(vmin, vmax, 50)
        marginal_kws.setdefault('bins', bins)
    if kind not in ('reg', 'resid'):
        joint_kws.setdefault('alpha', 0.5)

    jointgrid = sns.jointplot(sample1, sample2, joint_kws=joint_kws,
                              marginal_kws=marginal_kws, kind=kind, **kwargs)
    xmin, xmax, ymin, ymax = jointgrid.ax_joint.axis()

    xmin = max(xmin, sample1.min() - .1)
    ymin = max(ymin, sample2.min() - .1)
    jointgrid.ax_joint.set_xlim(xmin, xmax)
    jointgrid.ax_joint.set_ylim(ymin, ymax)

def barplot(data, color=None, x_order=None, ax=None, title='', **kwargs):
    sns.barplot(data, color=color, x_order=x_order, ax=ax, **kwargs)
    grouped = data.groupby(data)
    ax.set_title(title)
    sizes = grouped.size()
    percents = sizes / sizes.sum() * 100
    xs = ax.get_xticks()

    annotate_yrange_factor = 0.025
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # Reset ymax and ymin so there's enough room to see the annotation of
    # the top-most
    if ymax > 0:
        ymax += yrange * 0.1
    if ymin < 0:
        ymin -= yrange * 0.1
    ax.set_ylim(ymin, ymax)
    yrange = ymax - ymin

    offset_ = yrange * annotate_yrange_factor
    for x, modality in zip(xs, x_order):
        try:
            y = sizes[modality]
            offset = offset_ if y >= 0 else -1 * offset_
            verticalalignment = 'bottom' if y >= 0 else 'top'
            percent = percents[modality]
            ax.annotate('{} ({:.1f}%)'.format(y, percent),
                            (x, y + offset),
                            verticalalignment=verticalalignment,
                            horizontalalignment='center')
        except KeyError:
            ax.annotate('0 (0%)',
                            (x, offset_),
                            verticalalignment='bottom',
                            horizontalalignment='center')