"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


@pytest.fixture(params=[None, 10], ids=['n_none', 'n_10'])
def n(request):
    return request.param


class TestSplicingData:

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

    @pytest.fixture(params=['groupby_real', 'groupby_none'])
    def groupby_params(self, request, groupby_fixed):
        if request.param == 'groupby_real':
            return groupby_fixed
        elif request.param == 'groupby_none':
            return None

    # @pytest.fixture(params=[0.5, 12])
    # def min_samples(self, request):
    #     return request.param

    @pytest.fixture(params=[True, False])
    def percentages(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def rename(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def return_means(self, request):
        return request.param

    def test__subset_and_standardize(self, splicing):
        test_subset = splicing._subset_and_standardize(splicing.data)

        true_subset = splicing._subset(splicing.data)
        true_subset = true_subset.dropna(how='all', axis=1).dropna(how='all',
                                                                   axis=0)

        true_subset = true_subset.fillna(0.5)
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        pdt.assert_frame_equal(test_subset, true_subset)

    def test__subset_and_standardize_rename_means(self, splicing_fixed,
                                                  rename):
        test_subset, test_means = splicing_fixed._subset_and_standardize(
            splicing_fixed.data, return_means=True, rename=rename)

        true_subset = splicing_fixed._subset(splicing_fixed.data)
        true_subset = true_subset.dropna(how='all', axis=1).dropna(how='all',
                                                                   axis=0)

        true_subset = true_subset.fillna(0.5)
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        true_means = true_subset.mean()

        if rename:
            true_means = true_means.rename_axis(splicing_fixed.feature_renamer)
            true_subset = true_subset.rename_axis(
                splicing_fixed.feature_renamer, 1)

        pdt.assert_frame_equal(test_subset, true_subset)
        pdt.assert_series_equal(test_means, true_means)

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

            def thresh(x):
                return n * x.shape[0]

            test_positions = splicing.nmf_space_positions(groupby)
        else:
            test_positions = splicing.nmf_space_positions(groupby, n=n)

            def thresh(x):
                return n

        grouped = splicing.singles.groupby(groupby)
        at_least_n_per_group_per_event = pd.concat(
            [df.dropna(thresh=thresh(df), axis=1) for name, df in grouped])
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

    # @pytest.mark.parameterize('n_groups', 2)
    def test_big_nmf_space_transitions(self, splicing_fixed, groupby_fixed,
                                       group_transitions_fixed):
        test_big_transitions = splicing_fixed.big_nmf_space_transitions(
            groupby_fixed, group_transitions_fixed)

        nmf_space_transitions = splicing_fixed.nmf_space_transitions(
            groupby_fixed, group_transitions_fixed)

        # get the mean and standard dev of the whole array
        n = nmf_space_transitions.count().sum()
        mean = nmf_space_transitions.sum().sum() / n
        std = np.sqrt(np.square(nmf_space_transitions - mean).sum().sum() / n)

        true_big_transitions = nmf_space_transitions[
            nmf_space_transitions > (mean + std)].dropna(how='all')

        pdt.assert_frame_equal(test_big_transitions, true_big_transitions)

    # @pytest.mark.parameterize('n_groups', 2)
    def test_is_nmf_space_x_axis_included(self, splicing_fixed, groupby_fixed):
        test_is_nmf_space_x_axis_included = \
            splicing_fixed._is_nmf_space_x_axis_excluded(groupby_fixed)

        nmf_space_positions = splicing_fixed.nmf_space_positions(groupby_fixed)

        # Get the correct included/excluded labeling for the x and y axes
        event, phenotype = nmf_space_positions.pc_1.argmax()
        top_pc1_samples = splicing_fixed.data.groupby(groupby_fixed).groups[
            phenotype]

        data = splicing_fixed._subset(splicing_fixed.data,
                                      sample_ids=top_pc1_samples)
        binned = splicing_fixed.binify(data)
        true_is_nmf_space_x_axis_included = bool(binned[event][0])

        pdt.assert_equal(test_is_nmf_space_x_axis_included,
                         true_is_nmf_space_x_axis_included)

    # @pytest.mark.parameterize('n_groups', 2)
    def test_nmf_space_xlabel(self, splicing_fixed, groupby_fixed):
        test_xlabel = splicing_fixed._nmf_space_xlabel(groupby_fixed)

        if splicing_fixed._is_nmf_space_x_axis_excluded(groupby_fixed):
            true_xlabel = splicing_fixed.excluded_label
        else:
            true_xlabel = splicing_fixed.included_label

        pdt.assert_equal(test_xlabel, true_xlabel)

    # @pytest.mark.parameterize('n_groups', 2)
    def test_nmf_space_ylabel(self, splicing_fixed, groupby_fixed):
        test_ylabel = splicing_fixed._nmf_space_ylabel(groupby_fixed)

        if splicing_fixed._is_nmf_space_x_axis_excluded(groupby_fixed):
            true_ylabel = splicing_fixed.included_label
        else:
            true_ylabel = splicing_fixed.excluded_label

        pdt.assert_equal(test_ylabel, true_ylabel)

    # @pytest.mark.parameterize(n_groups=3)
    def test_plot_big_nmf_space(self, splicing_fixed,
                                groupby_fixed, group_to_color_fixed,
                                group_order_fixed, group_transitions_fixed,
                                color_ordered_fixed, group_to_marker):
        splicing_fixed.plot_big_nmf_space_transitions(
            groupby_fixed, group_transitions_fixed, group_order_fixed,
            color_ordered_fixed, group_to_color_fixed, group_to_marker)
        plt.close('all')

    def test_modality_assignments(self, splicing_fixed, groupby_params):
        sample_ids = None
        feature_ids = None
        test_modality_assignments = splicing_fixed.modality_assignments(
            sample_ids=sample_ids, feature_ids=feature_ids,
            groupby=groupby_params)

        data = splicing_fixed._subset(splicing_fixed.data, sample_ids,
                                      feature_ids, require_min_samples=False)
        if groupby_params is None:
            groupby_copy = pd.Series('all', index=data.index)
        else:
            groupby_copy = groupby_params

        grouped = data.groupby(groupby_copy)
        data = pd.concat([df.dropna(thresh=10, axis=1)
                         for name, df in grouped])
        true_assignments = data.groupby(groupby_copy).apply(
            splicing_fixed.modality_estimator.fit_transform)

        pdt.assert_frame_equal(test_modality_assignments, true_assignments)

    @pytest.mark.xfail
    def test_modality_assignments_all_inputs_not_none(self, splicing_fixed,
                                                      groupby_fixed):
        sample_ids = None
        feature_ids = None
        splicing_fixed.modality_assignments(
            sample_ids=sample_ids, feature_ids=feature_ids,
            data=splicing_fixed.singles,
            groupby=groupby_fixed)

    @pytest.mark.xfail
    def test_modality_assignments_invalid_thresh(self, splicing_fixed,
                                                 groupby_fixed):
        sample_ids = None
        feature_ids = None
        splicing_fixed.modality_assignments(
            sample_ids=sample_ids, feature_ids=feature_ids, min_samples=None,
            groupby=groupby_fixed)

    def test_modality_counts(self, splicing_fixed):
        sample_ids = None
        feature_ids = None
        test_modality_counts = splicing_fixed.modality_counts(
            sample_ids=sample_ids, feature_ids=feature_ids)

        assignments = splicing_fixed.modality_assignments(sample_ids,
                                                          feature_ids)
        true_counts = assignments.apply(lambda x: x.groupby(x).size(), axis=1)
        pdt.assert_frame_equal(test_modality_counts, true_counts)

    def test_plot_modalities_bars(self, splicing_fixed, groupby_fixed,
                                  group_to_color_fixed, percentages):
        splicing_fixed.plot_modalities_bars(
            groupby=groupby_fixed, percentages=percentages,
            phenotype_to_color=group_to_color_fixed)

    def test_plot_modalities_reduced(self, splicing_fixed, groupby_fixed,
                                     group_to_color_fixed):
        splicing_fixed.plot_modalities_bars(
            groupby=groupby_fixed, phenotype_to_color=group_to_color_fixed)

    def test_plot_modalities_lavalamps(self, splicing_fixed, groupby_fixed,
                                       group_to_color_fixed):
        splicing_fixed.plot_modalities_lavalamps(
            groupby=groupby_fixed, phenotype_to_color=group_to_color_fixed)

    def test_plot_feature(self, splicing_fixed):
        splicing_fixed.plot_feature(splicing_fixed.data.columns[0])

    def test_plot_lavalamp(self, splicing_fixed, group_to_color_fixed):
        splicing_fixed.plot_lavalamp(group_to_color_fixed)

    def test_plot_two_features(self, splicing_fixed, groupby_fixed,
                               group_to_color_fixed):
        ind = splicing_fixed.data.count() > 10

        features = splicing_fixed.data.columns[ind]
        feature1 = features[0]
        feature2 = features[1]
        splicing_fixed.plot_two_features(feature1, feature2,
                                         groupby=groupby_fixed,
                                         label_to_color=group_to_color_fixed)

    def test_plot_two_samples(self, splicing_fixed):
        samples = splicing_fixed.data.index[splicing_fixed.data.T.count() > 10]
        sample1 = samples[0]
        sample2 = samples[1]
        splicing_fixed.plot_two_samples(sample1, sample2)
