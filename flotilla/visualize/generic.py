from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from flotilla.visualize.splicing import plot_pooled_dot


def violinplot(psi, groupby=None, color=None, ax=None, pooled_psi=None,
               order=None, violinplot_kws=None, title=None,
               label_pooled=True, data_type='splicing'):
    splicing = 'splicing'.startswith(data_type)

    if ax is None:
        ax = plt.gca()

    violinplot_kws = {} if violinplot_kws is None else violinplot_kws

    # Add a tiny amount of random noise in case the values are all identical,
    # Otherwise we get a LinAlg error.
    psi += np.random.uniform(0, 0.001, psi.shape[0])

    inner = 'points' if splicing else 'box'
    sns.violinplot(psi, groupby=groupby, bw=0.1, inner=inner,
                   color=color, linewidth=0.5, order=order,
                   ax=ax, **violinplot_kws)
    if pooled_psi is not None:
        grouped = pooled_psi.groupby(groupby)
        if order is not None:
            for i, name in enumerate(order):
                try:
                    subset = pooled_psi.ix[grouped.groups[name]]
                    plot_pooled_dot(ax, subset, x_offset=i, label=label_pooled)
                except KeyError:
                    pass
        else:
            plot_pooled_dot(ax, pooled_psi)

    if splicing:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('$\Psi$')

    if title is not None:
        ax.set_title(title)
    sns.despine()