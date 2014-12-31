import numpy.testing as npt
import pandas.util.testing as pdt


class TestMappingStatsData(object):

    def test__init(self, mapping_stats_data, mapping_stats_kws):
        from flotilla.data_model.quality_control import MappingStatsData, \
            MIN_READS

        test_data = MappingStatsData(mapping_stats_data, **mapping_stats_kws)

        min_reads = mapping_stats_kws.pop('min_reads', MIN_READS)
        number_mapped_col = mapping_stats_kws.pop('number_mapped_col')

        npt.assert_equal(test_data.min_reads, min_reads)
        npt.assert_equal(test_data.number_mapped_col, number_mapped_col)

        number_mapped = mapping_stats_data[number_mapped_col]
        too_few_mapped = number_mapped.index[number_mapped < min_reads]

        pdt.assert_series_equal(test_data.number_mapped, number_mapped)
        pdt.assert_array_equal(test_data.too_few_mapped, too_few_mapped)