
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from scipy import stats
from ..util import timeout, TimeoutError


def switchy_score(array):
    """Transform a 1D array of data scores to a vector of "switchy scores"

    Calculates std deviation and mean of sine- and cosine-transformed
    versions of the array. Better than sorting by just the mean which doesn't
    push the really lowly variant events to the ends.

    Parameters
    ----------
    array : numpy.array
        A 1-D numpy array or something that could be cast as such (like a list)

    Returns
    -------
    float
        The "switchy score" of the study_data which can then be compared to other
        splicing event study_data

    @author Michael T. Lovci
    """
    array = np.array(array)
    variance = 1 - np.std(np.sin(array[~np.isnan(array)] * np.pi))
    mean_value = -np.mean(np.cos(array[~np.isnan(array)] * np.pi))
    return variance * mean_value

def get_switchy_score_order(x):
    """Apply switchy scores to a 2D array of data scores

    Parameters
    ----------
    x : numpy.array
        A 2-D numpy array in the shape [n_events, n_samples]

    Returns
    -------
    numpy.array
        A 1-D array of the ordered indices, in switchy score order
    """
    switchy_scores = np.apply_along_axis(switchy_score, axis=0, arr=x)
    return np.argsort(switchy_scores)


def binify(df, bins):
    """Makes a histogram of each row the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe whose rows you'd like to binify.
    bins : numpy.array
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1)

    Returns
    -------
    binned : pandas.DataFrame

    Raises
    ------


    """
    ncol = bins.shape[0] - 1
    nrow = df.shape[0]
    binned = np.zeros((nrow, ncol))

    # TODO: make sure this works for numpy matrices
    for i, (name, row) in enumerate(df.iterrows()):
        binned[i, :] = np.histogram(row, bins=bins, normed=True)[0]

    columns = ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]
    binned = pd.DataFrame(binned, index=df.index, columns=columns)
    return binned


def get_regressor(x,y, n_estimators=1500, pCut=0.05, n_tries=5, verbose=False):

    if verbose:
        sys.stderr.write('getting regressor\n')
    clfs = []
    oob_scores = []

    for i in range(n_tries):
        if verbose:
            sys.stderr.write('%d.' % i)

        clf = ExtraTreesRegressor(n_estimators=n_estimators, oob_score=True,
                                  bootstrap=True, max_features='sqrt',
                                  n_jobs=1, random_state=i).fit(x,y)
        clfs.append(clf)
        oob_scores.append(clf.oob_score_)
    clf = clfs[np.argmax(oob_scores)]
    clf.feature_importances = pd.Series(clf.feature_importances_, index=x.columns)

    return clf, oob_scores


def get_boosting_regressor(x,y,verbose=False):
    if verbose:
        sys.stderr.write('getting boosting regressor\n')

    clf = GradientBoostingRegressor(n_estimators=50, subsample=0.6, max_features=100,
                                    verbose=0, learning_rate=0.1, random_state=0).fit(x,y)

    clf.feature_importances = pd.Series(clf.feature_importances_, index=x.columns)
    if verbose:
        sys.stderr.write('finished boosting regressor\n')

    return clf


def get_unstarted_events(mongodb):
    """
    get events that have not been started yet.
    generator sets started to True before returning an event
    """
    go_on = True
    while go_on ==True:

        event = mongodb['list'].find_one({"started":False})

        if event is None:
            go_on=False

        else:
            event['started'] = True
            mongodb['list'].save(event)
            yield event

@timeout(5) #because these sometimes hang
def get_slope(x,y):
    return stats.linregress(x,y)[0]

@timeout(5) #because these sometimes hang
def do_r(s_1, s_2, method=stats.pearsonr, min_items=12):

    """
    do an R calculation, remove items with values missing in either

    input:
    x - predictor vector
    y - target vector

    optional:
    method - method to use (scipy.stats.pearsonr or scipy.stats.spearmanr)

    output:
    r, p (order as determined by the chosen method)

    return (nan, nan) if too few items overlap

    """
    s_1, s_2 = s_1.dropna().align(s_2.dropna(), join='inner')
    if len(s_1) <= min_items:
        return (np.nan, np.nan)
    return method(s_1, s_2)

@timeout(10) #because these sometimes hang
def get_robust_values(x,y):
    """
    get robust linear regression

    input:
    x - predictor vector
    y - target vector

    output:
    intercept, slope, t-statistic, p-value

    """
    import statsmodels.api as sm
    rlm_result = sm.RLM(y, sm.add_constant(x), missing='drop').fit()
    return rlm_result.params[0], rlm_result.params[1], rlm_result.tvalues[0], rlm_result.pvalues[0],

@timeout(5)
def get_dcor(x, y):
    """
    get dcor

    see: https://github.com/andrewdyates/dcor

    input:
    x - predictor vector
    y - target vector

    output:
    dc, dr, dvx, dvy

    """
    import dcor_cpy as dcor
    dc, dr, dvx, dvy = dcor.dcov_all(x,y)
    return dc, dr, dvx, dvy

@timeout(100)
def apply_calc_rs(X, y, method = stats.pearsonr):
    """
    apply R calculation method on each gene separately (for nan values)

    input:
    X - (cells X rpkms) pd.DataFrame
    y - (cells X psi) pd.Series

    output:
    two pd.Series of:
    R coef, p-value
    """

    out_R = pd.Series(index=X.columns, name=y.name)
    out_P = pd.Series(index=X.columns, name=y.name)
    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            r, p = do_r(x, y, method=method)

        except TimeoutError:
            sys.stderr.write("%s r timeout event:%s, gene:%s\n" % (method, y.name, x.name))
            r, p = np.nan, np.nan
        out_R.ix[this_id] = r
        out_P.ix[this_id] = p
    return out_R, out_P

@timeout(220)
def apply_calc_robust(X, y, verbose=False):

    """X and y are dataframes, returns slope, t-value and p-value of robust regression"""

    if verbose:
        sys.stderr.write("getting robust regression\n")
    out_I = pd.Series(index=X.columns, name=y.name) #intercept
    out_S = pd.Series(index=X.columns, name=y.name) #slope
    out_T = pd.Series(index=X.columns, name=y.name) #t-value
    out_P = pd.Series(index=X.columns, name=y.name) #p-value

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            i, s, t, p = get_robust_values(x, y)
        except TimeoutError:
            sys.stderr.write("robust timeout event:%s, gene:%s\n" % (y.name, x.name))
            i, s, t, p = np.nan, np.nan, np.nan, np.nan
        out_I.ix[this_id] = i
        out_S.ix[this_id] = s
        out_T.ix[this_id] = t
        out_P.ix[this_id] = p
    return out_I, out_S, out_T, out_P

@timeout(50)
def apply_calc_slope(X, y, verbose=False):
    """X and y are dataframes, returns slope, t-value and p-value of robust regression"""
    if verbose:
        sys.stderr.write("getting slope\n")

    out_S = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            s = get_slope(x, y)
        except TimeoutError:
            sys.stderr.write("linregress timeout event:%s, gene:%s\n" % (y.name, x.name))
            s = np.nan
        out_S.ix[this_id] = s

    return out_S

@timeout(50)
def apply_dcor(X, y, verbose=False):

    if verbose:
        sys.stderr.write("getting dcor\n")

    out_DC = pd.Series(index=X.columns, name=y.name)
    out_DR = pd.Series(index=X.columns, name=y.name)
    out_DVX = pd.Series(index=X.columns, name=y.name)
    out_DVY = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            dc, dr, dvx, dvy = get_dcor(*map(np.array, [x,y]))

        except TimeoutError:
            sys.stderr.write("dcor timeout event:%s, gene:%s\n" % (y.name, x.name))
            dc, dr, dvx, dvy = [np.nan] * 4
        out_DC.ix[this_id] = dc
        out_DR.ix[this_id] = dr
        out_DVX.ix[this_id] = dvx
        out_DVY.ix[this_id] = dvy
    return out_DC, out_DR, out_DVX, out_DVY

# <codecell>

def dropna_mean(x):
    return x.dropna().mean()
