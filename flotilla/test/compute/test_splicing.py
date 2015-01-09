from collections import Iterable

import pytest
import numpy as np
import numpy.testing as npt
from scipy import stats


class TestModalityModel(object):

    @pytest.fixture(params=[1, np.arange(1, 5)])
    def alphas(self, request):
        return request.param

    @pytest.fixture(params=[1, np.arange(1, 5)])
    def betas(self, request):
        return request.param

    def test_init(self, alphas, betas):
        from flotilla.compute.splicing import ModalityModel

        model = ModalityModel(alphas, betas)

        true_alphas = alphas if isinstance(alphas, Iterable) else np.ones(
            len(betas)) * alphas
        true_betas = betas if isinstance(betas, Iterable) else np.ones(
            len(alphas)) * betas

        true_rvs = [stats.beta(a, b) for a, b in
                    zip(true_alphas, true_betas)]
        true_scores = np.arange(len(true_rvs)).astype(float) + .1
        true_scaled_scores = true_scores / true_scores.max()
        true_prob_parameters = true_scaled_scores / true_scaled_scores.sum()

        npt.assert_array_equal(model.alphas, true_alphas)
        npt.assert_array_equal(model.betas, true_betas)
        npt.assert_array_equal(model.scores, true_scores)
        npt.assert_array_equal(model.scaled_scores, true_scaled_scores)
        npt.assert_array_equal(model.prob_parameters, true_prob_parameters)
        for test_rv, true_rv in zip(model.rvs, true_rvs):
            npt.assert_array_equal(test_rv.args, true_rv.args)


# class TestModalities:
#
#     def test_init(self, kwargs):
#         from flotilla.compute.splicing import Modalities
#
#         test_modalities = Modalities(**kwargs)
#
#         if kwargs == {}:
#             npt.assert_equal(test_modalities.bins, (0, 0.2, 0.8, 1))
#         else:
#             npt.assert_equal(test_modalities.bins, (0, kwargs['excluded_max'],
#                                                     kwargs['included_min'], 1))
#
# def test_binned_to_assignments():
#     from flotilla.compute.splicing import TRUE_MODALITIES, \
#         _binned_to_assignments
#
#     assignments = _binned_to_assignments(TRUE_MODALITIES, TRUE_MODALITIES)
#
#     npt.assert_array_equal(assignments.values, assignments.index)