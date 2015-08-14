"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


@pytest.fixture
def n():
    return 10


class TestSplicingData:

    @pytest.fixture(params=['groupby_real', 'groupby_none'])
    def groupby_params(self, request, groupby):
        if request.param == 'groupby_real':
            return groupby
        elif request.param == 'groupby_none':
            return None

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

        true_subset = true_subset.fillna(true_subset.mean())
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        pdt.assert_frame_equal(test_subset, true_subset)

    def test__subset_and_standardize_rename_means(self, splicing,
                                                  rename):
        test_subset, test_means = splicing._subset_and_standardize(
            splicing.data, return_means=True, rename=rename)

        true_subset = splicing._subset(splicing.data)
        true_subset = true_subset.dropna(how='all', axis=1).dropna(how='all',
                                                                   axis=0)

        true_subset = true_subset.fillna(true_subset.mean())
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        true_means = true_subset.mean()

        if rename:
            true_means = true_means.rename_axis(splicing.feature_renamer)
            true_subset = true_subset.rename_axis(
                splicing.feature_renamer, 1)

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
            n = 20

            def thresh(x):
                return n

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

        test = splicing.transition_distances(nmf_positions, group_transitions)

        nmf_positions.index = nmf_positions.index.droplevel(0)
        true = pd.Series(index=group_transitions)
        for transition in group_transitions:
            try:
                phenotype1, phenotype2 = transition
                norm = np.linalg.norm(
                    nmf_positions.ix[phenotype2] - nmf_positions.ix[
                        phenotype1])
                # print phenotype1, phenotype2, norm
                true[transition] = norm
            except KeyError:
                pass

        pdt.assert_series_equal(test, true)

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

    # @pytest.mark.parameterize('n_groups', 2)
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

    def test_plot_feature(self, splicing):
        splicing.plot_feature(splicing.data.columns[0])
        plt.close('all')

    def test_plot_lavalamp(self, splicing, group_to_color):
        splicing.plot_lavalamp(group_to_color)
        plt.close('all')

    def test_plot_two_features(self, splicing, groupby,
                               group_to_color):
        ind = splicing.data.count() > 10

        features = splicing.data.columns[ind]
        feature1 = features[0]
        feature2 = features[1]
        splicing.plot_two_features(feature1, feature2, groupby=groupby,
                                   label_to_color=group_to_color)
        plt.close('all')

    def test_plot_two_samples(self, splicing):
        samples = splicing.data.index[splicing.data.T.count() > 10]
        sample1 = samples[0]
        sample2 = samples[1]
        splicing.plot_two_samples(sample1, sample2)
        plt.close('all')
