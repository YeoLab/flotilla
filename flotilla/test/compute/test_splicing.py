from collections import Iterable

import pytest
import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.misc import logsumexp

class TestModalityModel(object):

    @pytest.fixture(params=[1, np.arange(1, 5)])
    def alphas(self, request):
        return request.param

    @pytest.fixture(params=[1, np.arange(1, 5)])
    def betas(self, request):
        return request.param
    
    @pytest.fixture()
    def alpha(self):
        return np.arange(1, 5)
    
    @pytest.fixture()
    def beta(self):
        return 1.
    
    @pytest.fixture()
    def x(self):
        return np.arange(0, 1.1, 0.1)

    @pytest.fixture()
    def model(self, alpha, beta):
        from flotilla.compute.splicing import ModalityModel
        return ModalityModel(alpha, beta)

    def test_init(self, alphas, betas):
        from flotilla.compute.splicing import ModalityModel

        model = ModalityModel(alphas, betas)

        true_alphas = alphas
        true_betas = betas
        if not isinstance(alphas, Iterable) and not isinstance(betas, Iterable):
            true_alphas = [alphas]
            true_betas = [betas]

        true_alphas = true_alphas if isinstance(true_alphas, Iterable) else np.ones(
            len(true_betas)) * true_alphas
        true_betas = true_betas if isinstance(true_betas, Iterable) else np.ones(
            len(true_alphas)) * true_betas

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

    def test_logliks(self, x, model):
        test_logliks = model.logliks(x)
        
        true_x = x.copy()
        true_x[true_x == 0] = 0.001
        true_x[true_x == 1] = 0.999
        true_logliks = np.array([np.log(prob) + rv.logpdf(true_x).sum()
                                 for prob, rv in zip(model.prob_parameters,
                                                     model.rvs)])
        npt.assert_array_equal(test_logliks, true_logliks)

    def test_logsumexp_logliks(self, x, model):
        test_logsumexp_logliks = model.logsumexp_logliks(x)

        npt.assert_array_equal(test_logsumexp_logliks,
                               logsumexp(model.logliks(x)))


# class TestModalityEstimator(object):
#     def test_init(self, splicing):