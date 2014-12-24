"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""
import copy

import numpy as np
import pytest


@pytest.fixture
def splicing_data(shalek2013_data, minimum_samples):
    from flotilla.data_model import SplicingData

    return SplicingData(shalek2013_data.splicing, minimum_samples=minimum_samples)

@pytest.fixture(params=[None, 100])
def data_for_binned_nmf_reduced(request, splicing_data):
    if request.param is None:
        return None
    else:
        psi = copy.deepcopy(splicing_data.data)
        max_index = np.prod(map(lambda x: x-1, psi.shape))
        random_flat_indices = np.random.randint(0, max_index, 100)
        psi.values[np.unravel_index(random_flat_indices, psi.shape)] = np.nan
        return psi

@pytest.fixture(params=[None, 10])
def n(request):
    return request.param

# class TestSplicingData:
#     def test_init(self, splicing_data, shalek2013_data):
#
#         if splicing_data.minimum_samples > 0:
#             if not splicing_data.singles.empty:
#                 singles = shalek2013_data.splicing.ix[splicing_data.singles.index]
#                 data = splicing_data._threshold(shalek2013_data.splicing,
#                                                 singles)
#             else:
#                 data = splicing_data._threshold(shalek2013_data.splicing)
#         else:
#             data = splicing_data.data_original
#
#         pdt.assert_frame_equal(splicing_data.data_original,
#                                shalek2013_data.splicing)
#         pdt.assert_frame_equal(splicing_data.data, data)
#
#     def test_binify(self, splicing_data):
#         from flotilla.compute.infotheory import binify
#         test_binned = splicing_data.binify(splicing_data.data)
#
#         true_binned = binify(splicing_data.data, splicing_data.bins)
#         true_binned = true_binned.dropna(how='all', axis=1)
#
#         pdt.assert_frame_equal(test_binned, true_binned)
#
#     def test_binned_nmf_reduced(self, splicing_data,
#                                 data_for_binned_nmf_reduced):
#         test_binned_nmf_reduced = splicing_data.binned_nmf_reduced(
#             data=data_for_binned_nmf_reduced)
#
#         if data_for_binned_nmf_reduced is None:
#             data = splicing_data.data
#         else:
#             data = data_for_binned_nmf_reduced
#
#         binned = splicing_data.binify(data)
#         true_binned_nmf_reduced = splicing_data.nmf.transform(binned.T)
#
#         pdt.assert_frame_equal(
#             test_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1),
#             true_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1))

    # Temporary commenting out while chr22 dataset is down
    # def test_nmf_space_positions(self, chr22, n):
    #     """Use chr22 dataset because need multiple phenotypes"""
    #     groupby = chr22.sample_id_to_phenotype
    #
    #     if n is None:
    #         test_positions = chr22.splicing.nmf_space_positions(groupby)
    #     else:
    #         test_positions = chr22.splicing.nmf_space_positions(groupby, n=n)
    #
    #     if n is None:
    #         n = 5
    #     grouped = chr22.splicing.singles.groupby(groupby)
    #     at_least_n_per_group_per_event = grouped.transform(
    #         lambda x: x if x.count() >= n
    #         else pd.Series(np.nan, index=x.index))
    #     df = at_least_n_per_group_per_event.groupby(groupby).apply(
    #         lambda x: chr22.splicing.binned_nmf_reduced(data=x))
    #     df = df.swaplevel(0, 1)
    #     true_positions = df.sort_index()
    #
    #     pdt.assert_frame_equal(test_positions, true_positions)
    #
    # def test_transition_distances(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #     nmf_positions = chr22.splicing.nmf_space_positions(groupby=groupby)
    #     transitions = chr22.phenotype_transitions
    #
    #     test_distances = chr22.splicing.transition_distances(nmf_positions,
    #                                                          transitions)
    #
    #     nmf_positions.index = nmf_positions.index.droplevel(0)
    #     true_distances = pd.Series(index=transitions)
    #     for transition in transitions:
    #         try:
    #             phenotype1, phenotype2 = transition
    #             norm = np.linalg.norm(nmf_positions.ix[phenotype2] - nmf_positions.ix[phenotype1])
    #             # print phenotype1, phenotype2, norm
    #             true_distances[transition] = norm
    #         except KeyError:
    #             pass
    #
    #     pdt.assert_series_equal(test_distances, true_distances)
    #
    # def test_nmf_space_transitions(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #     nmf_space_positions = chr22.splicing.nmf_space_positions(groupby=groupby)
    #     phenotype_transitions = chr22.phenotype_transitions
    #
    #     test_transitions = chr22.splicing.nmf_space_transitions(
    #         groupby, phenotype_transitions)
    #
    #     nmf_space_positions = nmf_space_positions.groupby(
    #         level=0, axis=0).filter(lambda x: len(x) > 1)
    #
    #     nmf_space_transitions = nmf_space_positions.groupby(
    #         level=0, axis=0, as_index=True, group_keys=False).apply(
    #         chr22.splicing.transition_distances,
    #         transitions=phenotype_transitions)
    #
    #     # Remove any events that didn't have phenotype pairs from
    #     # the transitions
    #     true_transitions = nmf_space_transitions.dropna(how='all', axis=0)
    #
    #     pdt.assert_frame_equal(test_transitions, true_transitions)
    #
    # def test_big_nmf_space_transitions(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #     phenotype_transitions = chr22.phenotype_transitions
    #
    #     test_big_transitions = chr22.splicing.big_nmf_space_transitions(
    #         groupby, phenotype_transitions)
    #
    #     nmf_space_transitions = chr22.splicing.nmf_space_transitions(
    #         groupby, phenotype_transitions)
    #
    #     # get the mean and standard dev of the whole array
    #     n = nmf_space_transitions.count().sum()
    #     mean = nmf_space_transitions.sum().sum() / n
    #     std = np.sqrt(np.square(nmf_space_transitions - mean).sum().sum() / n)
    #
    #     true_big_transitions = nmf_space_transitions[
    #         nmf_space_transitions > (mean + 2 * std)].dropna(how='all')
    #
    #     pdt.assert_frame_equal(test_big_transitions, true_big_transitions)
    #
    # def test_is_nmf_space_x_axis_included(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #
    #     test_is_nmf_space_x_axis_included = \
    #         chr22.splicing._is_nmf_space_x_axis_excluded(groupby)
    #
    #     nmf_space_positions = chr22.splicing.nmf_space_positions(groupby)
    #
    #     # Get the correct included/excluded labeling for the x and y axes
    #     event, phenotype = nmf_space_positions.pc_1.argmax()
    #     top_pc1_samples = chr22.splicing.data.groupby(groupby).groups[
    #         phenotype]
    #
    #     data = chr22.splicing._subset(chr22.splicing.data, sample_ids=top_pc1_samples)
    #     binned = chr22.splicing.binify(data)
    #     true_is_nmf_space_x_axis_included = bool(binned[event][0])
    #
    #     pdt.assert_equal(test_is_nmf_space_x_axis_included,
    #                      true_is_nmf_space_x_axis_included)
    #
    # def test_nmf_space_xlabel(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #     test_xlabel = chr22.splicing._nmf_space_xlabel(groupby)
    #
    #     if chr22.splicing._is_nmf_space_x_axis_excluded(groupby):
    #         true_xlabel = chr22.splicing.excluded_label
    #     else:
    #         true_xlabel = chr22.splicing.included_label
    #
    #     pdt.assert_equal(test_xlabel, true_xlabel)
    #
    # def test_nmf_space_ylabel(self, chr22):
    #     groupby = chr22.sample_id_to_phenotype
    #     test_ylabel = chr22.splicing._nmf_space_ylabel(groupby)
    #
    #     if chr22.splicing._is_nmf_space_x_axis_excluded(groupby):
    #         true_ylabel = chr22.splicing.included_label
    #     else:
    #         true_ylabel = chr22.splicing.excluded_label
    #
    #     pdt.assert_equal(test_ylabel, true_ylabel)
    #
    # def test_plot_big_nmf_space(self, chr22):
    #     chr22.splicing.plot_big_nmf_space_transitions(
    #         chr22.sample_id_to_phenotype,
    #         chr22.phenotype_transitions,
    #         chr22.phenotype_order,
    #         chr22.phenotype_color_ordered,
    #         chr22.phenotype_to_color,
    #         chr22.phenotype_to_marker)
    #     plt.close('all')