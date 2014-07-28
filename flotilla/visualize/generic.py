from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from flotilla.visualize.splicing import plot_pooled_dot


def violinplot(data, groupby=None, color=None, ax=None, pooled_data=None,
               order=None, violinplot_kws=None, title=None,
               label_pooled=True, data_type='splicing'):
    data_type = 'none' if data_type is None else data_type

    splicing = 'splicing'.startswith(data_type)

    if ax is None:
        ax = plt.gca()

    violinplot_kws = {} if violinplot_kws is None else violinplot_kws

    # Add a tiny amount of random noise in case the values are all identical,
    # Otherwise we get a LinAlg error.
    data += np.random.uniform(0, 0.001, data.shape[0])

    inner = 'points' if splicing else 'box'
    sns.violinplot(data, groupby=groupby, bw=0.1, inner=inner,
                   color=color, linewidth=0.5, order=order,
                   ax=ax, **violinplot_kws)
    if pooled_data is not None:
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

    if splicing:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('$\Psi$')

    if title is not None:
        ax.set_title(title)
    sns.despine()