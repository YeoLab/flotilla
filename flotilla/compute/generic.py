"""
generic
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from scipy import stats

from ..util import timeout, TimeoutError


def get_regressor(x, y, n_estimators=1500, n_tries=5,
                  verbose=False):
    """Calculate an ExtraTreesRegressor on predictor and target variables

    Parameters
    ----------
    x : numpy.array
        Predictor vector
    y : numpy.array
        Target vector
    n_estimators : int, optional
        Number of estimators to use
    n_tries : int, optional
        Number of attempts to calculate regression
    verbose : bool, optional
        If True, output progress statements

    Returns
    -------
    classifier : sklearn.ensemble.ExtraTreesRegressor
        The classifier with the highest out of bag scores of all the
        attempted "tries"
    oob_scores : numpy.array
        Out of bag scores of the classifier
    """
    if verbose:
        sys.stderr.write('Getting regressor\n')
    clfs = []
    oob_scores = []

    for i in range(n_tries):
        if verbose:
            sys.stderr.write('%d.' % i)

        clf = ExtraTreesRegressor(n_estimators=n_estimators, oob_score=True,
                                  bootstrap=True, max_features='sqrt',
                                  n_jobs=1, random_state=i).fit(x, y)
        clfs.append(clf)
        oob_scores.append(clf.oob_score_)
    clf = clfs[np.argmax(oob_scores)]
    clf.feature_importances = pd.Series(clf.feature_importances_,
                                        index=x.columns)

    return clf, oob_scores


def get_boosting_regressor(x, y, verbose=False):
    """Calculate a GradientBoostingRegressor on predictor and target variables

    Parameters
    ----------
    x : numpy.array
        Predictor variable
    y : numpy.array
        Target variable
    verbose : bool, optional
        If True, output status messages

    Returns
    -------
    classifier : sklearn.ensemble.GradientBoostingRegressor
        A fitted classifier of the predictor and target variable
    """
    if verbose:
        sys.stderr.write('Getting boosting regressor\n')

    clf = GradientBoostingRegressor(n_estimators=50, subsample=0.6,
                                    max_features=100,
                                    verbose=0, learning_rate=0.1,
                                    random_state=0).fit(x, y)

    clf.feature_importances = pd.Series(clf.feature_importances_,
                                        index=x.columns)
    if verbose:
        sys.stderr.write('Finished boosting regressor\n')

    return clf


def get_unstarted_events(mongodb):
    """
    get events that have not been started yet.
    generator sets started to True before returning an event

    Parameters
    ----------
    mongodb : pymongo.Database
        A MongoDB database object
    """
    go_on = True
    while go_on:
        event = mongodb['list'].find_one({"started": False})

        if event is None:
            go_on = False
        else:
            event['started'] = True
            mongodb['list'].save(event)
            yield event


@timeout(5)  # because these sometimes hang
def get_slope(x, y):
    """Get the linear regression slope of x and y

    Parameters
    ----------
    x : numpy.array
        X-values of data
    y : numpy.array
        Y-values of data

    Returns
    -------
    slope : float
        Scipy.stats.linregress slope

    """
    return stats.linregress(x, y)[0]


@timeout(5)  # because these sometimes hang
def do_r(s_1, s_2, method=stats.pearsonr, min_items=12):
    """Calculate correlation ("R-value") between two vectors

    Parameters
    ----------
    s_1 : pandas.Series
        Predictor vector
    s_2 : pandas.Series
        Target vector
    method : function, optional
        Which correlation method to use. (default scipy.stats.pearsonr)
    min_items : int, optional
        Minimum number of items occuring in both s_1 and s_2 (default 12)

    Returns
    -------
    r_value : float
        R-value of the correlation, i.e. how correlated the two inputs are
    p_value : float
        p-value of the correlation, i.e. how likely this correlation would
        happen given the null hypothesis that the two are not correlated

    Notes
    -----
    If too few items overlap, return (np.nan, np.nan)
    """
    s_1, s_2 = s_1.dropna().align(s_2.dropna(), join='inner')
    if len(s_1) <= min_items:
        return np.nan, np.nan
    return method(s_1, s_2)


@timeout(10)  # because these sometimes hang
def get_robust_values(x, y):
    """Calculate robust linear regression

    Parameters
    ----------
    x : numpy.array
        Predictor vector
    y : numpy.array
        Target vector

    Returns
    -------
    intercept : float
        Intercept of the fitted line
    slope : float
        Slope of the fitted line
    t_statistic : float
        T-statistic of the fit
    p_value : float
        p-value of the fit
    """
    import statsmodels.api as sm

    r = sm.RLM(y, sm.add_constant(x), missing='drop').fit()
    results = r.params[0], r.params[1], r.tvalues[0], r.pvalues[0]
    return results


@timeout(5)
def get_dcor(x, y):
    """Calculate distance correlation between two vectors

    Uses the distance correlation package from:
    https://github.com/andrewdyates/dcor

    Parameters
    ----------
    x : numpy.array
        1-dimensional array (aka a vector) of the independent, predictor
        variable
    y : numpy.array
        1-dimensional array (aka a vector) of the dependent, target variable

    Returns
    -------
    dc : float
        Distance covariance
    dr : float
        Distance correlation
    dvx : float
        Distance variance on x
    dvy : float
        Distance variance on y
    """
    # cython version of dcor
    try:
        import dcor_cpy as dcor
    except ImportError as e:
        sys.stderr.write("Please install dcor_cpy.")
        raise e

    dc, dr, dvx, dvy = dcor.dcov_all(x, y)
    return dc, dr, dvx, dvy


@timeout(100)
def apply_calc_rs(X, y, method=stats.pearsonr):
    """Apply R calculation method on each column of X versus the values of y

    Parameters
    ----------
    X : pandas.DataFrame
        A (n_samples, n_features) sized DataFrame, assumed to be of
        log-normal expression values
    y : pandas.Series
        A (n_samples,) sized Series, assumed to be of percent spliced-in
        alternative splicing scores
    method : function, optional
        Which correlation method to use on each feature in X versus the
        values in y

    Returns
    -------
    r_coefficients : pandas.Series
        Correlation coefficients
    p_values : pandas.Series
        Correlation significances (smaller is better)

    See Also
    --------
    do_r
        This is the underlying function which calculates correlation
    """
    out_R = pd.Series(index=X.columns, name=y.name)
    out_P = pd.Series(index=X.columns, name=y.name)
    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            r, p = do_r(x, y, method=method)

        except TimeoutError:
            sys.stderr.write(
                "%s r timeout event:%s, gene:%s\n" % (method, y.name, x.name))
            r, p = np.nan, np.nan
        out_R.ix[this_id] = r
        out_P.ix[this_id] = p
    return out_R, out_P


@timeout(220)
def apply_calc_robust(X, y, verbose=False):
    """Calculate robust regression between the columns of X and y

    Parameters
    ----------
    X : pandas.DataFrame
        A (n_samples, n_features) Dataframe of the predictor variable
    y : pandas.DataFrame
        A (n_samples, m_features) DataFrame of the response variable
    verbose : bool, optional
        If True, output status messages as the calculation is happening

    Returns
    -------
    out_I : pandas.Series
        Intercept of regressions
    out_S : pandas.Series
        Slope of regressions
    out_T : pandas.Series
        t-statistic of regressions
    out_P : pandas.Series
        p-values of regressions

    See Also
    --------
    get_robust_values
        This is the underlying function which calculates the slope,
        intercept, t-value, and p-value of the fit
    """
    if verbose:
        sys.stderr.write("getting robust regression\n")
    out_I = pd.Series(index=X.columns, name=y.name)  # intercept
    out_S = pd.Series(index=X.columns, name=y.name)  # slope
    out_T = pd.Series(index=X.columns, name=y.name)  # t-value
    out_P = pd.Series(index=X.columns, name=y.name)  # p-value

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            i, s, t, p = get_robust_values(x, y)
        except TimeoutError:
            sys.stderr.write(
                "robust timeout event:%s, gene:%s\n" % (y.name, x.name))
            i, s, t, p = np.nan, np.nan, np.nan, np.nan
        out_I.ix[this_id] = i
        out_S.ix[this_id] = s
        out_T.ix[this_id] = t
        out_P.ix[this_id] = p
    return out_I, out_S, out_T, out_P


@timeout(50)
def apply_calc_slope(X, y, verbose=False):
    """X and y are dataframes, returns slope, t-value and p-value of robust
    regression

    Parameters
    ----------
    X : pandas.DataFrame
        A (n_samples, n_features) Dataframe of predictor variable values
    y : pandas.DataFrame
        A (n_samples, m_features) Dataframe of response variable values
    verbose : bool, optional
        If True, output status messages

    Returns
    -------
    slope : pandas.Series
        Slopes of the linear regression

    See Also
    --------
    get_slope
        This is the underlying function which calculates the slope
    """
    if verbose:
        sys.stderr.write("getting slope\n")

    out_S = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            s = get_slope(x, y)
        except TimeoutError:
            sys.stderr.write(
                "linregress timeout event:%s, gene:%s\n" % (y.name, x.name))
            s = np.nan
        out_S.ix[this_id] = s

    return out_S


@timeout(50)
def apply_dcor(X, y, verbose=False):
    """Calcualte distance correlation between the columns of two dataframes

    Parameters
    ----------
    X : pandas.DataFrame
        A (n_samples, n_features) Dataframe of predictor variable values
    y : pandas.DataFrame
        A (n_samples, m_features) Dataframe of response variable values
    verbose : bool, optional
        If True, output status messages

    Returns
    -------
    dc : pandas.Series
        Distance covariance
    dr : pandas.Series
        Distance correlation
    dvx : pandas.Series
        Distance variance of x
    dvy : pandas.Series
        Distance variance of y

    See Also
    --------
    get_dcor
        This is the underlying function that gets called to calculate the
        distance correlation
    """
    if verbose:
        sys.stderr.write("getting dcor\n")

    out_DC = pd.Series(index=X.columns, name=y.name)
    out_DR = pd.Series(index=X.columns, name=y.name)
    out_DVX = pd.Series(index=X.columns, name=y.name)
    out_DVY = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            dc, dr, dvx, dvy = get_dcor(*map(np.array, [x, y]))

        except TimeoutError:
            sys.stderr.write("dcor timeout event:%s, gene:%s\n" % (y.name,
                                                                   x.name))
            dc, dr, dvx, dvy = [np.nan] * 4
        out_DC.ix[this_id] = dc
        out_DR.ix[this_id] = dr
        out_DVX.ix[this_id] = dvx
        out_DVY.ix[this_id] = dvy
    return out_DC, out_DR, out_DVX, out_DVY


def dropna_mean(x):
    """Drop NA values and return the mean
    """
    return x.dropna().mean()


def spearmanr_series(x, y):
    """Calculate spearman r (with p-values) between two pandas series

    Parameters
    ----------
    x : pandas.Series
        One of the two series you'd like to correlate
    y : pandas.Series
        The other series you'd like to correlate

    Returns
    -------
    r_value : float
        The R-value of the correlation. 1 for perfect positive correlation,
        and -1 for perfect negative correlation
    p_value : float
        The p-value of the correlation.
    """
    x, y = x.dropna().align(y.dropna(), 'inner')
    return stats.spearmanr(x, y)


def spearmanr_dataframe(A, B, axis=0):
    """Calculate spearman correlations between dataframes A and B

    Parameters
    ----------
    A : pandas.DataFrame
        A n_samples x n_features1 dataframe. Must have the same number of rows
        as "B"
    B : pandas.DataFrame
        A n_samples x n_features2 Dataframe. Must have the same number of rows
        as "A"
    axis : int
        Which axis to compare. If 0, calculate correlations between all the
        columns of A vs te columns of B. If 1, calculate between rows.
        (default 0)

    Returns
    -------
    correlations : pandas.DataFrame
        A n_features2 x n_features1 DataFrame of (spearman_r, spearman_p)
        tuples

    Notes
    -----
    Use "applymap" to get just the R- and p-values of the resulting dataframe

    >>> import pandas as pd
    >>> import numpy as np
    >>> A = pd.DataFrame(np.random.randn(100).reshape(5, 20))
    >>> B = pd.DataFrame(np.random.randn(55).reshape(5, 11))
    >>> correls = spearmanr_dataframe(A, B)
    >>> correls.shape
    (11, 20)
    >>> spearman_r = correls.applymap(lambda x: x[0])
    >>> spearman_p = correls.applymap(lambda x: x[1])
    """
    return A.apply(lambda x: B.apply(lambda y: spearmanr_series(x, y),
                                     axis=axis),
                   axis=axis)
