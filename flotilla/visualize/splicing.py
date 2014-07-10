import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..compute.splicing import get_switchy_score_order
from .color import red, blue, purple, grey, green


class ModalitiesViz(object):
    """Visualize results of modality assignments
    """
    modalities_colors = {'included': red,
                         'excluded': blue,
                         'bimodal': purple,
                         'uniform': grey,
                         'middle': green}

    modalities_order = ['excluded', 'uniform', 'bimodal', 'middle',
                        'included']

    colors = [modalities_colors[modality] for modality in
              modalities_order]

    def plot_reduced_space(self, binned_reduced, modalities_assignments,
                           ax=None, title=None):
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
        """Plot a single splicing event's changes in NMF space, and its
        violin plots

        """
        pass


def lavalamp(psi, color=None, x_offset=0, title='', ax=None):
    """Make a 'lavalamp' scatter plot of many splicing events

    Useful for visualizing many splicing events at once.

    Parameters
    ----------
    TODO.md: (n_events, n_samples).transpose()
    data : array
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


    Returns
    -------
    fig : matplotlib.Figure
        A figure object for saving.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))
    else:
        fig = plt.gcf()

    try:
        # This is a pandas Dataframe
        y = psi.values
    except AttributeError:
        # This is a numpy array
        y = psi

    order = get_switchy_score_order(y)
    y = y[:, order]

    n_samples, n_events = psi.shape
    # .astype(float) is to get rid of a deprecation warning
    # x = np.repeat(np.arange(n_samples), n_events).reshape(n_samples,
    #                                                        n_events).astype(float)
    x = np.vstack((np.arange(n_events) for _ in xrange(n_samples))).astype(
        float)
    x += x_offset

    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1

    color = green if color is None else color
    ax.plot(x, y, 'd', color=color, alpha=0.2, markersize=10)
    sns.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(n_events))
    ax.set_xticks([])

    ax.set_xlim(-0.5, xmax + .5)
    ax.set_ylim(0, 1)
    ax.set_title(title)
