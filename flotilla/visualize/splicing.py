import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..compute.splicing import get_switchy_score_order
from .color import red, blue, purple, grey, green
import itertools


class ModalitiesViz(object):
    """Visualize results of modality assignments
    
    Attributes
    ----------
    
    
    Methods
    -------
    
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

    def __init__(self):
        """Constructor for ModalitiesViz
        
        Parameters
        ----------
        assignments, binned
        
        Returns
        -------
        
        
        Raises
        ------
        
        """
        # super(ModalitiesViz, self).__init__(binned.T, n_components=2)
        # self.modalities_assignments = modalities_assignments
        # self.binned_reduced = binned_reduced

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

def lavalamp(psi, color=None, jitter=None, title='', ax=None):
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
        fig, ax = plt.subplots(figsize=(16,4))
    else:
        fig = plt.gcf()
    nrow, ncol = psi.shape
    x = np.vstack(np.arange(ncol,) for i in range(nrow)).T

    color = pd.Series('#FFFF00', index=psi.index) if color is None else color

    try:
        # This is a pandas Dataframe
        y = psi.values
    except AttributeError:
        # This is a numpy array
        y = psi

    if jitter is None:
        jitter = np.ones(y.shape[0])
    else:
        assert np.all(np.abs(jitter) < 1)
        assert np.min(jitter) > -.0000000001

    order = get_switchy_score_order(y)
    y = y[:, order]
    assert type(color) == pd.Series
    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1

    x_jitter = np.array([i+jitter for i in x]).T

    for co, xx, yy in zip(color, x_jitter, y):

        ax.scatter(xx, yy, marker='d', color=co, alpha=0.2, edgecolor='none', linewidth=0.5, )
        #print map(int, xx), map(lambda x: "%.2f" % x, yy)
        #import pdb
        #pdb.set_trace()#

    sns.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(ncol))
    ax.set_xticks([])

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_title(title)