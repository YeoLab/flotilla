from collections import defaultdict
from itertools import cycle

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import pandas.util.testing as pdt


class TestMetaData(object):
    @pytest.fixture
    def phenotype_col(self):
        return 'phenotype'

    @pytest.fixture
    def n(self):
        return 20

    @pytest.fixture
    def metadata(self, n, phenotype_col):
        index = ['sample_{}'.format(i + 1) for i in range(n)]
        metadata = pd.DataFrame(index=index)
        metadata[phenotype_col] = np.random.choice(list('ABC'), size=20)
        metadata['subset1'] = np.random.choice([True, False], size=20)
        return metadata

    @pytest.fixture(params=[None, list('CAB')])
    def phenotype_order(self, request):
        return request.param

    @pytest.fixture(params=[None, {'A': 'blue', 'B': 'red', 'C': 'green'},
                            {'A': '#f66ab5', 'B': '#f77189', 'C': '#6cad31'}])
    def phenotype_to_color(self, request):
        return request.param

    @pytest.fixture(params=[None, {'A': '*', 'B': 'o', 'C': 's'}])
    def phenotype_to_marker(self, request):
        return request.param

    @pytest.fixture
    def kws(self, phenotype_col, phenotype_order, phenotype_to_color,
            phenotype_to_marker):
        return {'minimum_sample_subset': 2,
                'phenotype_col': phenotype_col,
                'phenotype_to_marker': phenotype_to_marker,
                'phenotype_order': phenotype_order,
                'phenotype_to_color': phenotype_to_color}

    def test_init(self, metadata, phenotype_col,
                  phenotype_order, phenotype_to_color,
                  phenotype_to_marker, kws):
        from flotilla.data_model.metadata import MetaData
        from flotilla.data_model.base import subsets_from_metadata
        from flotilla.visualize.color import str_to_color

        test_metadata = MetaData(metadata, **kws)

        if phenotype_order is None:
            true_phenotype_order = list(sorted(
                test_metadata.unique_phenotypes))
        else:
            true_phenotype_order = phenotype_order

        if phenotype_to_color is None:
            default_phenotype_to_color = \
                test_metadata._default_phenotype_to_color
            true_phenotype_to_color = dict(
                (k, default_phenotype_to_color[k])
                for k in true_phenotype_order)
        else:
            true_phenotype_to_color = {}
            for phenotype, color in phenotype_to_color.iteritems():
                try:
                    color = str_to_color[color]
                except KeyError:
                    pass
                true_phenotype_to_color[phenotype] = color

        if phenotype_to_marker is None:
            markers = cycle(['o', '^', 's', 'v', '*', 'D', ])

            def marker_factory():
                return markers.next()
            true_phenotype_to_marker = defaultdict(marker_factory)
            for x in true_phenotype_order:
                true_phenotype_to_marker[x]

        else:
            true_phenotype_to_marker = phenotype_to_marker

        true_phenotype_transitions = zip(true_phenotype_order[:-1],
                                         true_phenotype_order[1:])
        true_unique_phenotypes = metadata[phenotype_col].unique()
        true_n_phenotypes = len(true_unique_phenotypes)

        true_colors = map(mpl.colors.rgb2hex,
                          sns.color_palette('husl',
                                            n_colors=true_n_phenotypes))
        colors = iter(true_colors)
        true_default_phenotype_to_color = defaultdict(lambda: colors.next())

        true_sample_id_to_phenotype = metadata[phenotype_col]
        true_phenotype_color_order = [true_phenotype_to_color[p]
                                      for p in true_phenotype_order]
        true_sample_id_to_color = \
            dict((i, true_phenotype_to_color[true_sample_id_to_phenotype[i]])
                 for i in metadata.index)

        true_sample_subsets = subsets_from_metadata(
            metadata, kws['minimum_sample_subset'], 'samples')

        pdt.assert_frame_equal(test_metadata.data, metadata)
        pdt.assert_series_equal(test_metadata.sample_id_to_phenotype,
                                true_sample_id_to_phenotype)
        pdt.assert_array_equal(test_metadata.unique_phenotypes,
                               true_unique_phenotypes)
        pdt.assert_array_equal(test_metadata.n_phenotypes,
                               len(true_unique_phenotypes))
        pdt.assert_array_equal(test_metadata._default_phenotype_order,
                               list(sorted(true_unique_phenotypes)))
        pdt.assert_array_equal(test_metadata.phenotype_order,
                               true_phenotype_order)
        pdt.assert_array_equal(test_metadata.phenotype_transitions,
                               true_phenotype_transitions)
        pdt.assert_array_equal(test_metadata._colors, true_colors)
        pdt.assert_array_equal(test_metadata._default_phenotype_to_color,
                               true_default_phenotype_to_color)
        pdt.assert_dict_equal(test_metadata.phenotype_to_color,
                              true_phenotype_to_color)
        pdt.assert_dict_equal(test_metadata.phenotype_to_marker,
                              true_phenotype_to_marker)
        pdt.assert_array_equal(test_metadata.phenotype_color_order,
                               true_phenotype_color_order)
        pdt.assert_dict_equal(test_metadata.sample_id_to_color,
                              true_sample_id_to_color)
        pdt.assert_dict_equal(test_metadata.sample_subsets,
                              true_sample_subsets)

    def test_ignore_subset_columns(self, metadata, phenotype_order,
                                   phenotype_col,
                                   phenotype_to_color, phenotype_to_marker,
                                   kws):
        from flotilla.data_model.metadata import MetaData

        metadata = metadata.copy()
        metadata['no_subset'] = np.arange(metadata.shape[0])
        ignore_subset_columns = 'no_subset'

        test_metadata = MetaData(metadata, **kws)
        assert 'no_subset' not in test_metadata.sample_subsets

    def test_change_phenotype_col(self, metadata, n, phenotype_col,
                                  phenotype_order, phenotype_to_color,
                                  phenotype_to_marker, kws):
        from flotilla.data_model.metadata import MetaData

        metadata = metadata.copy()
        metadata['phenotype2'] = np.random.choice(list('QXYZ'), size=n)

        test_metadata = MetaData(metadata, **kws)
        test_metadata.phenotype_col = 'phenotype2'

        pdt.assert_array_equal(test_metadata.unique_phenotypes,
                               metadata.phenotype2.unique())
        pdt.assert_contains_all(metadata.phenotype2.unique(),
                                test_metadata.phenotype_to_color)
        pdt.assert_contains_all(metadata.phenotype2.unique(),
                                test_metadata.phenotype_to_marker)
        pdt.assert_array_equal(test_metadata.phenotype_order,
                               list(sorted(metadata.phenotype2.unique())))
