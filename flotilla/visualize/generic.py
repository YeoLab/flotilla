from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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

    if not isinstance(groupby, pd.Series):
        groupby = pd.Series(groupby)

    if groupby.name is None:
        groupby.name = 'phenotype'
    tidy_singles = singles.dropna().to_frame().join(groupby)
    tidy_singles = tidy_singles.reset_index()
    tidy_singles = tidy_singles.rename(columns={singles.name: ylabel})

    if pooled is not None:
        tidy_pooled = pooled.dropna().to_frame().join(groupby)
        tidy_pooled = tidy_pooled.reset_index()
        tidy_pooled = tidy_pooled.rename(columns={pooled.name: ylabel})
    else:
        tidy_pooled = None

    if outliers is not None:
        tidy_outliers = outliers.dropna().to_frame().join(groupby)
        tidy_outliers = tidy_outliers.reset_index()
        tidy_outliers = tidy_outliers.rename(columns={outliers.name: ylabel})
    else:
        tidy_outliers = None

    if outliers is not None and not outliers.dropna().empty:
        sns.violinplot(x=groupby.name, y=ylabel, data=tidy_outliers,
                       bw=bw, order=order, inner=None, cut=0,
                       linewidth=1, scale='width', color='lightgrey', ax=ax,
                       **kwargs)
    if not singles.dropna().empty:
        sns.violinplot(x=groupby.name, y=ylabel, data=tidy_singles,
                       bw=bw, order=order, inner=None, cut=0,
                       linewidth=1, scale='width', palette=color_ordered,
                       ax=ax, **kwargs)
    if outliers is not None and not outliers.dropna().empty:
        sns.stripplot(x=groupby.name, y=ylabel, data=tidy_outliers,
                      jitter=True, order=order, ax=ax, color='grey')
    if not singles.dropna().empty:
        sns.stripplot(x=groupby.name, y=ylabel, data=tidy_singles,
                      jitter=True, order=order, ax=ax, palette=color_ordered)
    if pooled is not None and not pooled.dropna().empty:
        sns.stripplot(x=groupby.name, y=ylabel, data=tidy_pooled,
                      jitter=True, order=order, ax=ax, color='#262626',
                      zorder=100, size=10)
    sizes = tidy_singles.groupby(groupby.name).size()
    if order is None:
        order = sizes.keys()

    ax.set_xticklabels(['{0}\nn={1}'.format(group, sizes[group])
                        if group in sizes else group for group in order])

    to_set = {'yticks': yticks, 'ylim': ylim}
    to_set = {k: v for k, v in to_set.items() if v is not None}
    ax.set(**to_set)
    ax.set_title(title, fontsize=10)
    return ax


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
