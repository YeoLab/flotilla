from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

np.random.seed(0)


@pytest.fixture
def bins():
    return (0, .1, .2, .3, .4, 0.5, .6, .7, .8, .9, 1)


@pytest.fixture
def df1():
    return pd.DataFrame(np.random.uniform(size=200).reshape(10, 20))


@pytest.fixture
def df2():
    return pd.DataFrame(np.random.uniform(size=200).reshape(10, 20))


@pytest.fixture
def p(df1, bins):
    from flotilla.compute.infotheory import binify

    return binify(df1, bins)


@pytest.fixture
def q(df2, bins):
    from flotilla.compute.infotheory import binify

    return binify(df2, bins)


def test_bin_range_strings(bins):
    from flotilla.compute.infotheory import bin_range_strings

    bin_ranges = bin_range_strings(bins)
    true_bin_ranges = ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]
    npt.assert_equal(bin_ranges, true_bin_ranges)


def test_binify(bins, df1):
    from flotilla.compute.infotheory import bin_range_strings, binify

    binned = binify(df1, bins)

    true_binned = df1.apply(lambda x: pd.Series(np.histogram(x, bins=bins)[0]))
    true_binned.index = bin_range_strings(bins)
    true_binned = true_binned / true_binned.sum().astype(float)

    pdt.assert_frame_equal(binned, true_binned)


def test_kld(p, q):
    from flotilla.compute.infotheory import kld

    result = kld(p, q)

    p = p.replace(0, np.nan)
    q = q.replace(0, np.nan)
    true_result = (np.log2(p / q) * p).sum(axis=0)

    pdt.assert_series_equal(result, true_result)


def test_jsd(p, q):
    from flotilla.compute.infotheory import jsd, kld

    result = jsd(p, q)

    weight = 0.5
    m = weight * (p + q)
    true_result = weight * kld(p, m) + (1 - weight) * kld(q, m)

    pdt.assert_series_equal(result, true_result)


@pytest.fixture(params=[None, 2, 10])
def base(request):
    return request.param


def test_entropy(p, base):
    from flotilla.compute.infotheory import entropy

    if base is not None:
        result = entropy(p, base=base)
    else:
        result = entropy(p)
        base = 2

    true_result = -((np.log(p) / np.log(base)) * p).sum(axis=0)

    pdt.assert_series_equal(result, true_result)
