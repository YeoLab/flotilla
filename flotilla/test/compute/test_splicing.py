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

    @pytest.fixture
    def positive_control(self):
        """Randomly generated positive controls for modality estimation"""
        size = 20
        psi0 = pd.Series(np.random.uniform(0, 0.1, size=size), name='Psi~0')
        psi1 = pd.Series(np.random.uniform(0.9, 1, size=size), name='Psi~1')
        middle = pd.Series(np.random.uniform(0.45, 0.55, size=size),
                           name='middle')
        bimodal = pd.Series(np.concatenate([
            np.random.uniform(0, 0.1, size=size / 2),
            np.random.uniform(0.9, 1, size=size / 2)]), name='bimodal')
        df = pd.concat([psi0, psi1, middle, bimodal], axis=1)
        return df

    def test_init(self, step, vmax, logbf_thresh):
        from flotilla.compute.splicing import ModalityEstimator, \
            ModalityModel

        estimator = ModalityEstimator(step, vmax, logbf_thresh)

        true_parameters = np.arange(2, vmax + step, step).astype(float)
        true_exclusion = ModalityModel(1, true_parameters)
        true_inclusion = ModalityModel(true_parameters, 1)
        true_middle = ModalityModel(true_parameters+3, true_parameters+3)
        true_bimodal = ModalityModel(1 / (true_parameters+3),
                                     1 / (true_parameters+3))
        true_one_param_models = {'Psi~1': true_inclusion,
                                 'Psi~0': true_exclusion}
        true_two_param_models = {'bimodal': true_bimodal,
                                 'middle': true_middle}

        npt.assert_equal(estimator.step, step)
        npt.assert_equal(estimator.vmax, vmax)
        npt.assert_equal(estimator.logbf_thresh, logbf_thresh)
        npt.assert_equal(estimator.parameters, true_parameters)
        npt.assert_equal(estimator.exclusion_model, true_exclusion)
        npt.assert_equal(estimator.inclusion_model, true_inclusion)
        npt.assert_equal(estimator.middle_model, true_middle)
        npt.assert_equal(estimator.bimodal_model, true_bimodal)
        pdt.assert_dict_equal(estimator.one_param_models,
                              true_one_param_models)
        pdt.assert_dict_equal(estimator.two_param_models,
                              true_two_param_models)

    def test_fit_transform(self, estimator, splicing_data):
        test = estimator.fit_transform(splicing_data)

        # Estimate Psi~0/Psi~1 first (only one parameter change with each
        # paramterization)
        logbf_one_param = estimator._fit_transform_one_step(
            splicing_data, estimator.one_param_models)

        # Take everything that was below the threshold for included/excluded
        # and estimate bimodal and middle (two parameters change in each
        # parameterization
        ind = (logbf_one_param < estimator.logbf_thresh).all()
        ambiguous_columns = ind[ind].index
        data2 = splicing_data.ix[:, ambiguous_columns]
        logbf_two_param = estimator._fit_transform_one_step(
            data2, estimator.two_param_models)
        log2_bayes_factors = pd.concat([logbf_one_param, logbf_two_param],
                                       axis=0)

        # Make sure the returned dataframe has the same number of columns
        empty = splicing_data.count() == 0
        empty_columns = empty[empty].index
        empty_df = pd.DataFrame(np.nan, index=log2_bayes_factors.index,
                                columns=empty_columns)
        true = pd.concat([log2_bayes_factors, empty_df], axis=1)

        pdt.assert_frame_equal(test, true)

    @pytest.mark.xfail
    def test_fit_transform_greater_than1(self, estimator):
        nrows = 10
        ncols = 5
        data = pd.DataFrame(
            np.abs(np.random.randn(nrows, ncols).reshape(nrows, ncols))+10)
        estimator.fit_transform(data)

    @pytest.mark.xfail
    def test_fit_transform_less_than1(self, estimator):
        nrows = 10
        ncols = 5
        data = pd.DataFrame(
            np.abs(np.random.randn(nrows, ncols).reshape(nrows, ncols))-10)
        estimator.fit_transform(data)

    def test_assign_modalities(self, estimator, splicing_data):
        log2bf = estimator.fit_transform(splicing_data)
        test = estimator.assign_modalities(log2bf)

        x = log2bf
        not_na = (x.notnull() > 0).any()
        not_na_columns = not_na[not_na].index
        x.ix['ambiguous', not_na_columns] = estimator.logbf_thresh
        true = x.idxmax()

        pdt.assert_series_equal(test, true)

    def test_positive_control(self, estimator, positive_control):
        log2bf = estimator.fit_transform(positive_control)
        test = estimator.assign_modalities(log2bf)

        pdt.assert_almost_equal(test.values, test.index)


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
