from collections import Iterable

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
from scipy import stats
from scipy.misc import logsumexp


class TestModalityModel(object):
    @pytest.fixture()
    def x(self):
        return np.arange(0, 1.1, 0.1)

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
    def model(self, alpha, beta):
        from flotilla.compute.splicing import ModalityModel

        return ModalityModel(alpha, beta)

    def test_init(self, alphas, betas):
        from flotilla.compute.splicing import ModalityModel

        model = ModalityModel(alphas, betas)

        true_alphas = alphas
        true_betas = betas
        if not isinstance(alphas, Iterable) and not isinstance(betas,
                                                               Iterable):
            true_alphas = [alphas]
            true_betas = [betas]

        true_alphas = np.array(true_alphas) \
            if isinstance(true_alphas, Iterable) else np.ones(
            len(true_betas)) * true_alphas
        true_betas = np.array(true_betas) \
            if isinstance(true_betas, Iterable) else np.ones(
            len(true_alphas)) * true_betas

        true_rvs = [stats.beta(a, b) for a, b in
                    zip(true_alphas, true_betas)]
        true_scores = np.ones(true_alphas.shape).astype(float)
        true_scores = true_scores / true_scores.max()
        true_prob_parameters = true_scores / true_scores.sum()

        npt.assert_array_equal(model.alphas, true_alphas)
        npt.assert_array_equal(model.betas, true_betas)
        npt.assert_array_equal(model.scores, true_scores)
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

    def test_eq(self, alphas, betas):
        from flotilla.compute.splicing import ModalityModel

        model1 = ModalityModel(alphas, betas)
        model2 = ModalityModel(alphas, betas)
        assert model1 == model2

    def test_ne(self, alphas, betas):
        from flotilla.compute.splicing import ModalityModel

        if np.all(alphas == betas):
            assert 1
            return

        model1 = ModalityModel(alphas, betas)
        model2 = ModalityModel(betas, alphas)
        assert model1 != model2


class TestModalityEstimator(object):
    @pytest.fixture
    def step(self):
        return 1

    @pytest.fixture
    def vmax(self):
        return 10

    @pytest.fixture(params=[2, 3])
    def logbf_thresh(self, request):
        return request.param

    @pytest.fixture
    def estimator(self, step, vmax):
        from flotilla.compute.splicing import ModalityEstimator

        return ModalityEstimator(step, vmax)

    @pytest.fixture(params=['no_na', 'with_na'])
    def event(self, request):
        x = np.arange(0, 1.1, .1)
        if request.param == 'no_na':
            return x
        elif request.param == 'with_na':
            x[x < 0.5] = np.nan
            return x

    def test_init(self, step, vmax, logbf_thresh):
        from flotilla.compute.splicing import ModalityEstimator, \
            ModalityModel

        estimator = ModalityEstimator(step, vmax, logbf_thresh)

        true_parameters = np.arange(2, vmax + step, step).astype(float)
        true_exclusion = ModalityModel(1, true_parameters)
        true_inclusion = ModalityModel(true_parameters, 1)
        true_middle = ModalityModel(true_parameters+3, true_parameters+3)
        true_bimodal = ModalityModel(1 / (true_parameters+3),
                                     1 / (true_parameters+3),
                                     prior='exponential')
        true_models = {'included': true_inclusion,
                       'excluded': true_exclusion,
                       'bimodal': true_bimodal,
                       'middle': true_middle}

        npt.assert_equal(estimator.step, step)
        npt.assert_equal(estimator.vmax, vmax)
        npt.assert_equal(estimator.logbf_thresh, logbf_thresh)
        npt.assert_equal(estimator.parameters, true_parameters)
        npt.assert_equal(estimator.exclusion_model, true_exclusion)
        npt.assert_equal(estimator.inclusion_model, true_inclusion)
        npt.assert_equal(estimator.middle_model, true_middle)
        npt.assert_equal(estimator.bimodal_model, true_bimodal)
        pdt.assert_dict_equal(estimator.models, true_models)

    def test_loglik(self, event, estimator):
        test_loglik = estimator._loglik(event)

        true_loglik = dict((name, m.logliks(event))
                           for name, m in estimator.models.items())
        pdt.assert_dict_equal(test_loglik, true_loglik)

    def test_logsumexp(self, event, estimator):
        logliks = estimator._loglik(event)
        test_logsumexp = estimator._logsumexp(logliks)

        true_logsumexp = pd.Series(
            dict((name, logsumexp(loglik))
                 for name, loglik in logliks.items()))
        true_logsumexp['uniform'] = estimator.logbf_thresh
        pdt.assert_series_equal(test_logsumexp, true_logsumexp)

    def test_guess_modality(self, event, estimator):
        logsumexps = estimator._logsumexp(estimator._loglik(event))

        test_guess_modality = estimator._guess_modality(logsumexps)

        logsumexps['uniform'] = estimator.logbf_thresh
        true_guess_modality = logsumexps.idxmax()

        pdt.assert_equal(test_guess_modality, true_guess_modality)

    def test_fit_transform_with_na(self, estimator, splicing_data):
        test_fit_transform = estimator.fit_transform(splicing_data)

        logsumexp_logliks = splicing_data.apply(
            lambda x: pd.Series({k: v.logsumexp_logliks(x)
                                 for k, v in estimator.models.items()}),
            axis=0)
        logsumexp_logliks.ix['uniform'] = estimator.logbf_thresh
        true_fit_transform = logsumexp_logliks.idxmax()

        pdt.assert_series_equal(test_fit_transform, true_fit_transform)

    def test_fit_transform_no_na(self, estimator, splicing_data):
        test_fit_transform = estimator.fit_transform(splicing_data)

        logsumexp_logliks = splicing_data.apply(
            lambda x: pd.Series({k: v.logsumexp_logliks(x)
                                 for k, v in estimator.models.items()}),
            axis=0)
        logsumexp_logliks.ix['uniform'] = estimator.logbf_thresh
        true_fit_transform = logsumexp_logliks.idxmax()

        pdt.assert_series_equal(test_fit_transform, true_fit_transform)


@pytest.fixture(params=['list', 'array', 'nan'])
def array(request):
    x = np.arange(0, 1.1, .1)
    if request.param == 'list':
        return list(x)
    elif request.param == 'array':
        return x
    elif request.param == 'nan':
        x[x > .8] = np.nan
        return x


def test_switchy_score(array):
    from flotilla.compute.splicing import switchy_score

    test_switchy_score = switchy_score(array)

    true_array = np.array(array)
    variance = 1 - np.std(np.sin(true_array[~np.isnan(true_array)] * np.pi))
    mean_value = -np.mean(np.cos(true_array[~np.isnan(true_array)] * np.pi))
    true_switchy_score = variance * mean_value

    npt.assert_array_equal(test_switchy_score, true_switchy_score)


def test_get_switchy_score_order(splicing_data):
    from flotilla.compute.splicing import get_switchy_score_order, \
        switchy_score

    test_score_order = get_switchy_score_order(splicing_data)

    switchy_scores = np.apply_along_axis(switchy_score, axis=0,
                                         arr=splicing_data)
    true_score_order = np.argsort(switchy_scores)

    npt.assert_array_equal(test_score_order, true_score_order)
