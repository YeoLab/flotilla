from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
import seaborn as sns


@pytest.fixture
def pca(df_norm):
    from flotilla.compute.decomposition import DataFramePCA

    return DataFramePCA(df_norm)


@pytest.fixture
def kwargs():
    return dict(feature_renamer=None, groupby=None,
                singles=None, pooled=None, outliers=None,
                featurewise=False,
                order=None, violinplot_kws=None,
                data_type='expression', label_to_color=None,
                label_to_marker=None,
                scale_by_variance=True, x_pc='pc_1',
                y_pc='pc_2', n_vectors=20, distance='L1',
                n_top_pc_features=50, max_char_width=30)


@pytest.fixture
def large_dataframe():
    nrow = 100
    ncol = 1000
    index = ['sample_{}'.format(i) for i in np.arange(nrow)]
    columns = ['feature_{}'.format(i) for i in np.arange(ncol)]
    df = pd.DataFrame(data=np.random.randn(nrow, ncol), index=index,
                      columns=columns)

    return df


@pytest.fixture
def pca_large_dataframe(large_dataframe):
    from flotilla.compute.decomposition import DataFramePCA

    return DataFramePCA(large_dataframe)


def test_init(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, **kwargs)
    x_pc = kwargs['x_pc']
    y_pc = kwargs['y_pc']
    pcs = [x_pc, y_pc]

    true_groupby = dict.fromkeys(pca.reduced_space.index, 'all')
    true_grouped = pca.reduced_space.groupby(true_groupby, axis=0)

    colors = iter(sns.color_palette('husl',
                                    n_colors=len(true_grouped)))

    def color_factory():
        return colors.next()

    true_label_to_color = defaultdict(color_factory)

    markers = cycle(['o', '^', 's', 'v', '*', 'D', 'h'])

    def marker_factory():
        return markers.next()

    true_label_to_marker = defaultdict(marker_factory)
    for group in true_grouped.groups:
        true_label_to_marker[group]
    for group in dv.grouped.groups:
        dv.label_to_marker[group]

    true_color_ordered = [true_label_to_color[x] for x in
                          true_grouped.groups]

    true_vars = pca.explained_variance_ratio_[[x_pc, y_pc]]

    true_loadings = pca.components_.ix[[x_pc, y_pc]]
    true_loadings = true_loadings.multiply(true_vars, axis=0)

    reduced_space = pca.reduced_space[[x_pc, y_pc]]
    farthest_sample = reduced_space.apply(np.linalg.norm, axis=0).max()
    whole_space = true_loadings.apply(np.linalg.norm).max()
    scale = .25 * farthest_sample / whole_space
    true_loadings *= scale

    ord = 2 if kwargs['distance'] == 'L2' else 1
    true_magnitudes = true_loadings.apply(np.linalg.norm, ord=ord)
    true_magnitudes.sort(ascending=False)

    true_top_features = set([])
    true_pc_loadings_labels = {}
    true_pc_loadings = {}

    for pc in pcs:
        x = pca.components_.ix[pc].copy()
        x.sort(ascending=True)
        half_features = int(kwargs['n_top_pc_features'] / 2)
        if len(x) > kwargs['n_top_pc_features']:
            a = x[:half_features]
            b = x[-half_features:]
            labels = np.r_[a.index, b.index]
            true_pc_loadings[pc] = np.r_[a, b]
        else:
            labels = x.index
            true_pc_loadings[pc] = x

        true_pc_loadings_labels[pc] = labels
        true_top_features.update(labels)

    pdt.assert_frame_equal(dv.reduced_space, pca.reduced_space)
    pdt.assert_frame_equal(dv.components_, pca.components_)
    pdt.assert_series_equal(dv.explained_variance_ratio_,
                            pca.explained_variance_ratio_)
    pdt.assert_dict_equal(dv.label_to_marker, true_label_to_marker)
    pdt.assert_dict_equal(dv.groupby, true_groupby)
    pdt.assert_series_equal(dv.vars, true_vars)
    pdt.assert_frame_equal(dv.loadings, true_loadings)
    npt.assert_array_equal(dv.color_ordered, true_color_ordered)
    pdt.assert_series_equal(dv.magnitudes, true_magnitudes)
    npt.assert_array_equal(dv.top_features, true_top_features)
    pdt.assert_dict_equal(dv.pc_loadings_labels, true_pc_loadings_labels)
    pdt.assert_dict_equal(dv.pc_loadings, true_pc_loadings)


def test_large_dataframe(pca_large_dataframe, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca_large_dataframe.reduced_space,
                          pca_large_dataframe.components_,
                          pca_large_dataframe.explained_variance_ratio_,
                          **kwargs)
    x_pc = kwargs['x_pc']
    y_pc = kwargs['y_pc']
    pcs = [x_pc, y_pc]

    true_top_features = set([])
    true_pc_loadings_labels = {}
    true_pc_loadings = {}

    for pc in pcs:
        x = pca_large_dataframe.components_.ix[pc].copy()
        x.sort(ascending=True)
        half_features = int(kwargs['n_top_pc_features'] / 2)
        if len(x) > kwargs['n_top_pc_features']:
            a = x[:half_features]
            b = x[-half_features:]
            labels = np.r_[a.index, b.index]
            true_pc_loadings[pc] = np.r_[a, b]
        else:
            labels = x.index
            true_pc_loadings[pc] = x

        true_pc_loadings_labels[pc] = labels
        true_top_features.update(labels)
    pdt.assert_array_equal(dv.top_features, true_top_features)
    pdt.assert_dict_equal(dv.pc_loadings_labels, true_pc_loadings_labels)
    pdt.assert_dict_equal(dv.pc_loadings, true_pc_loadings)


def test_order(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    kw = kwargs.copy()
    kw.pop('order')
    kw.pop('groupby')
    groups = ['group1', 'group2', 'group3']

    groupby = pd.Series([np.random.choice(groups)
                         for i in pca.reduced_space.index],
                        index=pca.reduced_space.index)
    order = ['group3', 'group1', 'group2']

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, order=order,
                          groupby=groupby, **kw)

    color_ordered = [dv.label_to_color[x] for x in order]

    pdt.assert_series_equal(dv.groupby, groupby)
    pdt.assert_array_equal(dv.order, order)
    pdt.assert_array_equal(dv.color_ordered, color_ordered)


def test_explained_variance_none(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          None)
    true_vars = pd.Series([1., 1.], index=[kwargs['x_pc'], kwargs['y_pc']])
    pdt.assert_series_equal(dv.vars, true_vars)


def test_plot_samples(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, **kwargs)
    dv.plot_samples()
    ax = plt.gca()
    pdt.assert_equal(len(ax.lines), kwargs['n_vectors'] + 1)
    plt.close('all')


def test_plot_loadings(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, **kwargs)
    dv.plot_loadings()
    ax = plt.gca()
    pdt.assert_equal(len(ax.collections), 1)
    plt.close('all')


def test_plot(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, **kwargs)
    dv.plot()

    pdt.assert_equal(len(dv.fig_reduced.axes), 5)
    pdt.assert_equal(len(dv.ax_components.lines),
                     kwargs['n_vectors']+1)
    pdt.assert_equal(len(dv.ax_explained_variance.lines), 1)
    pdt.assert_equal(len(dv.ax_explained_variance.collections), 1)
    pdt.assert_equal(len(dv.ax_empty.collections), 0)
    pdt.assert_equal(len(dv.ax_pcs_heatmap.collections), 1)
    pdt.assert_equal(len(dv.ax_pcs_colorbar.collections), 1)
    assert not hasattr(dv, 'ax_loading1')
    assert not hasattr(dv, 'ax_loading2')
    plt.close('all')


def test_plot_loadings_scatter(pca, kwargs):
    from flotilla.visualize.decomposition import DecompositionViz

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_, **kwargs)
    dv.plot(plot_loadings='scatter')

    pdt.assert_equal(len(dv.fig_reduced.axes), 3)
    pdt.assert_equal(len(dv.ax_loading1.collections), 1)
    pdt.assert_equal(len(dv.ax_loading1.collections), 1)
    plt.close('all')


def test_plot_violins(pca, kwargs, df_norm):
    from flotilla.visualize.decomposition import DecompositionViz

    kw = kwargs.copy()
    kw.pop('singles')

    dv = DecompositionViz(pca.reduced_space, pca.components_,
                          pca.explained_variance_ratio_,
                          singles=df_norm, **kw)
    dv.plot(plot_violins=True)

    ncols = 4
    nrows = 1
    top_features = pd.Index(dv.top_features)
    vector_labels = list(set(dv.magnitudes[:dv.n_vectors].index.union(
        top_features)))
    while ncols * nrows < len(vector_labels):
        nrows += 1

    pdt.assert_equal(len(dv.fig_violins.axes), nrows * ncols)

    # for i in np.arange(len(top_features)):
    #     ax = dv.fig_violins.axes[i]
    #     pdt.assert_equal(len(ax.collections), len(dv.grouped.groups))
    plt.close('all')
