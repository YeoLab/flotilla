from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy.testing as npt
import pandas.util.testing as pdt
import pytest


class TestMappingStatsData(object):
    @pytest.fixture
    def mapping_stats(self, mapping_stats_data, mapping_stats_kws):
        from flotilla.data_model.quality_control import MappingStatsData

        return MappingStatsData(mapping_stats_data, **mapping_stats_kws)

    def test__init(self, mapping_stats, mapping_stats_kws):
        from flotilla.data_model.quality_control import MIN_READS

        min_reads = mapping_stats_kws.get('min_reads', MIN_READS)
        number_mapped_col = mapping_stats_kws.get('number_mapped_col')

        npt.assert_equal(mapping_stats.min_reads, min_reads)
        npt.assert_equal(mapping_stats.number_mapped_col, number_mapped_col)

    def test_number_mapped(self, mapping_stats, mapping_stats_data,
                           mapping_stats_kws):
        number_mapped_col = mapping_stats_kws.get('number_mapped_col')
        number_mapped = mapping_stats_data[number_mapped_col]
        pdt.assert_series_equal(mapping_stats.number_mapped, number_mapped)

    def test_too_few_mapped(self, mapping_stats, mapping_stats_data,
                            mapping_stats_kws):
        from flotilla.data_model.quality_control import MIN_READS

        min_reads = mapping_stats_kws.get('min_reads', MIN_READS)
        number_mapped_col = mapping_stats_kws.get('number_mapped_col')
        number_mapped = mapping_stats_data[number_mapped_col]
        too_few_mapped = number_mapped.index[number_mapped < min_reads]
        pdt.assert_numpy_array_equal(mapping_stats.too_few_mapped,
                                     too_few_mapped)
