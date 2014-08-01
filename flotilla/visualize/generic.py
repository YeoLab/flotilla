from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


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

    # Check that all the groups are represented, if not, add some data out of
    # range to the missing group
    if groupby is not None and order is not None:
        validated_groups = data.groupby(groupby).size().keys()
        verified_order = [x for x in order if x in validated_groups]
        positions = [i for i, x in enumerate(order) if x in validated_groups]
    else:
        verified_order = order
        positions = None

    inner = 'points' if splicing else 'box'
    sns.violinplot(data, groupby=groupby, bw=0.1, inner=inner,
                   color=color, linewidth=0.5, order=verified_order,
                   ax=ax, positions=positions, **violinplot_kws)
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

    if groupby is not None and order is not None:
        sizes = data.groupby(groupby).size()
        xticks = range(len(order))
        xticklabels = ['{}\nn={}'.format(group, sizes[group])
                       if group in sizes else '{}\nn=0'.format(group)
                       for group in order]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    sns.despine()


def plot_pooled_dot(ax, pooled, x_offset=0, label=True):
    try:
        xs = np.ones(pooled.shape[0])
    except AttributeError:
        xs = np.ones(1)
    xs += x_offset
    ax.plot(xs, pooled, 'o', color='#262626')

    if label:
        for x, y in zip(xs, pooled):
            if np.isnan(y):
                continue
            ax.annotate('pooled', (x, y), textcoords='offset points',
                        xytext=(7, 0), fontsize=14)