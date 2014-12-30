"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


@pytest.fixture(params=[None, 10])
def n(request):
    return request.param


class TestSplicingData:

    @pytest.fixture
    def splicing(self, splicing_data):
        from flotilla.data_model.splicing import SplicingData

        return SplicingData(splicing_data)

    # @pytest.fixture(params=[None, 100])
    # def data_for_binned_nmf_reduced(self, request, splicing):
    #     if request.param is None:
    #         return None
    #     else:
    #         psi = copy.deepcopy(splicing.data)
    #         max_index = np.prod(map(lambda x: x - 1, psi.shape))
    #         random_flat_indices = np.random.randint(0, max_index, 100)
    #         psi.values[
    #             np.unravel_index(random_flat_indices, psi.shape)] = np.nan
    #         return psi

    # def test_modality_assignments(self, splicing, groupby, true_modalities):
    #     assignments = splicing.modalities(groupby=groupby)
    #
    #     pdt.assert_frame_equal(assignments.sort_index(axis=1),
    #                            true_modalities.sort_index(axis=1))

    def test_binify(self, splicing):
        from flotilla.compute.infotheory import binify

        test_binned = splicing.binify(splicing.data)

        true_binned = binify(splicing.data, splicing.bins)
        true_binned = true_binned.dropna(how='all', axis=1)

        pdt.assert_frame_equal(test_binned, true_binned)

    def test_binned_nmf_reduced(self, splicing):
        test_binned_nmf_reduced = splicing.binned_nmf_reduced()

        data = splicing.data
        binned = splicing.binify(data)
        true_binned_nmf_reduced = splicing.nmf.transform(binned.T)

        pdt.assert_frame_equal(
            test_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1),
            true_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1))

    def test_nmf_space_positions(self, splicing, groupby, n):
        if n is None:
            n = 0.5
            test_positions = splicing.nmf_space_positions(groupby)
        else:
            test_positions = splicing.nmf_space_positions(groupby, n=n)

        grouped = splicing.singles.groupby(groupby)
        at_least_n_per_group_per_event = grouped.transform(
            lambda x: x if x.count() >= n
            else pd.Series(np.nan, index=x.index))
        df = at_least_n_per_group_per_event.groupby(groupby).apply(
            lambda x: splicing.binned_nmf_reduced(data=x))
        df = df.swaplevel(0, 1)
        true_positions = df.sort_index()

        pdt.assert_frame_equal(test_positions, true_positions)

    def test_transition_distances(self, splicing, groupby, group_transitions):
        nmf_positions = splicing.nmf_space_positions(groupby=groupby)

        test_distances = splicing.transition_distances(nmf_positions,
                                                             group_transitions)

        nmf_positions.index = nmf_positions.index.droplevel(0)
        true_distances = pd.Series(index=group_transitions)
        for transition in group_transitions:
            try:
                phenotype1, phenotype2 = transition
                norm = np.linalg.norm(
                    nmf_positions.ix[phenotype2] - nmf_positions.ix[
                        phenotype1])
                # print phenotype1, phenotype2, norm
                true_distances[transition] = norm
            except KeyError:
                pass

        pdt.assert_series_equal(test_distances, true_distances)

    def test_nmf_space_transitions(self, splicing, groupby, group_transitions):
        nmf_space_positions = splicing.nmf_space_positions(
            groupby=groupby)

        test_transitions = splicing.nmf_space_transitions(
            groupby, group_transitions)

        nmf_space_positions = nmf_space_positions.groupby(
            level=0, axis=0).filter(lambda x: len(x) > 1)

        nmf_space_transitions = nmf_space_positions.groupby(
            level=0, axis=0, as_index=True, group_keys=False).apply(
            splicing.transition_distances,
            transitions=group_transitions)

        # Remove any events that didn't have phenotype pairs from
        # the transitions
        true_transitions = nmf_space_transitions.dropna(how='all', axis=0)

        pdt.assert_frame_equal(test_transitions, true_transitions)

    def test_big_nmf_space_transitions(self, splicing, groupby, group_transitions):
        test_big_transitions = splicing.big_nmf_space_transitions(
            groupby, group_transitions)

        nmf_space_transitions = splicing.nmf_space_transitions(
            groupby, group_transitions)

        # get the mean and standard dev of the whole array
        n = nmf_space_transitions.count().sum()
        mean = nmf_space_transitions.sum().sum() / n
        std = np.sqrt(np.square(nmf_space_transitions - mean).sum().sum() / n)

        true_big_transitions = nmf_space_transitions[
            nmf_space_transitions > (mean + 2 * std)].dropna(how='all')

        pdt.assert_frame_equal(test_big_transitions, true_big_transitions)

    def test_is_nmf_space_x_axis_included(self, splicing, groupby):
        test_is_nmf_space_x_axis_included = \
            splicing._is_nmf_space_x_axis_excluded(groupby)

        nmf_space_positions = splicing.nmf_space_positions(groupby)

        # Get the correct included/excluded labeling for the x and y axes
        event, phenotype = nmf_space_positions.pc_1.argmax()
        top_pc1_samples = splicing.data.groupby(groupby).groups[
            phenotype]

        data = splicing._subset(splicing.data,
                                      sample_ids=top_pc1_samples)
        binned = splicing.binify(data)
        true_is_nmf_space_x_axis_included = bool(binned[event][0])

        pdt.assert_equal(test_is_nmf_space_x_axis_included,
                         true_is_nmf_space_x_axis_included)

    def test_nmf_space_xlabel(self, splicing, groupby):
        test_xlabel = splicing._nmf_space_xlabel(groupby)

        if splicing._is_nmf_space_x_axis_excluded(groupby):
            true_xlabel = splicing.excluded_label
        else:
            true_xlabel = splicing.included_label

        pdt.assert_equal(test_xlabel, true_xlabel)

    def test_nmf_space_ylabel(self, splicing, groupby):
        test_ylabel = splicing._nmf_space_ylabel(groupby)

        if splicing._is_nmf_space_x_axis_excluded(groupby):
            true_ylabel = splicing.included_label
        else:
            true_ylabel = splicing.excluded_label

        pdt.assert_equal(test_ylabel, true_ylabel)

    def test_plot_big_nmf_space(self, splicing, groupby, group_transitions,
                                group_order, group_to_color,
                                color_ordered, group_to_marker):
        splicing.plot_big_nmf_space_transitions(
            groupby, group_transitions, group_order, color_ordered,
            group_to_color, group_to_marker)
        plt.close('all')