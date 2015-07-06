from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def violinplot(singles, groupby, color_ordered=None, ax=None,
               pooled=None, ylabel='', bw=None,
               order=None, title=None, ylim=None, yticks=None,
               outliers=None, **kwargs):
    """
    Parameters
    ----------
    singles : pandas.Series
        Gene expression or splicing values from single cells
    groupby : pandas.Series
        A sample to group mapping for each sample id
    color_ordered : list, optional
        List of colors in the correct phenotype order
    ax : matplotlib.Axes object, optional
        Axes to plot on
    pooled : pandas.Series
        Gene expression or splicing values from pooled samples
    ylabel : str
        How to label the y-axis
    bw : float
        Width of the bandwidths for estimating the kernel density
    order : list
        Which order to plot the groups in
    title : str
        Title of the plot
    ylim : tuple
        Length 2 tuple specifying the minimum and maxmimum values to
        be plotted on the y-axis
    yticks : list
        Where to position yticks
    outliers : pandas.Series
        Gene expression or splicing values from outlier cells

    Returns
    -------
    ax : matplotlib.Axes object
        Axes with violinplot plotted
    """
    if ax is None:
        ax = plt.gca()

    tidy_singles = singles.dropna().to_frame().join(groupby)
    tidy_singles = tidy_singles.reset_index()
    tidy_singles = tidy_singles.rename(columns={singles.name: ylabel})

    if pooled is not None:
        tidy_pooled = pooled.dropna().to_frame().join(groupby)
        tidy_pooled = tidy_pooled.reset_index()
        tidy_pooled = tidy_pooled.rename(columns={pooled.name: ylabel})

    if outliers is not None:
        tidy_outliers = outliers.dropna().to_frame().join(groupby)
        tidy_outliers = tidy_outliers.reset_index()
        tidy_outliers = tidy_outliers.rename(columns={outliers.name: ylabel})

    if outliers is not None and not outliers.dropna().empty:
        sns.violinplot(x='phenotype', y=ylabel, data=tidy_outliers,
                       bw=bw, order=order, inner=None, cut=0,
                       linewidth=1, scale='width', color='lightgrey', ax=ax,
                       **kwargs)
    if not singles.dropna().empty:
        sns.violinplot(x='phenotype', y=ylabel, data=tidy_singles,
                       bw=bw, order=order, inner=None, cut=0,
                       linewidth=1, scale='width', palette=color_ordered,
                       ax=ax,
                       **kwargs)
    if outliers is not None and not outliers.dropna().empty:
        sns.stripplot(x='phenotype', y=ylabel, data=tidy_outliers,
                      jitter=True, order=order, ax=ax, color='grey')
    if not singles.dropna().empty:
        sns.stripplot(x='phenotype', y=ylabel, data=tidy_singles,
                      jitter=True, order=order, ax=ax, palette=color_ordered)
    if pooled is not None and not pooled.dropna().empty:
        sns.stripplot(x='phenotype', y=ylabel, data=tidy_pooled,
                      jitter=True, order=order, ax=ax, color='#262626')
    sizes = tidy_singles.groupby('phenotype').size()

    ax.set_xticklabels(['{0}\nn={1}'.format(group, sizes[group])
                        if group in sizes else group for group in order])
    ax.set(title=title, yticks=yticks, ylim=ylim)
    return ax


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
