"""
Information-theoretic calculations
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation

def bin_range_strings(bins):
    """Given a list of bins, make a list of strings of those bin ranges

    Parameters
    ----------
    bins : list_like
        List of anything, usually values of bin edges

    Returns
    -------
    bin_ranges : list
        List of bin ranges

    >>> bin_range_strings((0, 0.5, 1))
    ['0-0.5', '0.5-1']
    """
    return ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]


def binify(df, bins):
    """Makes a histogram of each column the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        A samples x features dataframe. Each feature (column) will be binned
        into the provided bins
    bins : iterable
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1).
        nbins = len(bins) - 1

    Returns
    -------
    binned : pandas.DataFrame
        An nbins x features DataFrame of each column binned across rows
    """
    if bins is None:
        raise ValueError('Must specify "bins"')
    binned = df.apply(lambda x: pd.Series(np.histogram(x, bins=bins)[0]))
    binned.index = bin_range_strings(bins)

    # Normalize so each column sums to 1
    binned = binned / binned.sum().astype(float)
    return binned


def kld(p, q):
    """Kullback-Leiber divergence of two probability distributions pandas
    dataframes, p and q

    Parameters
    ----------
    p : pandas.DataFrame
        An nbins x features DataFrame
    q : pandas.DataFrame
        An nbins x features DataFrame

    Returns
    -------
    kld : pandas.Series
        Kullback-Lieber divergence of the common columns between the
        dataframe. E.g. between 1st column in p and 1st column in q, and 2nd
        column in p and 2nd column in q.

    Notes
    -----
    The input to this function must be probability distributions, not raw
    values. Otherwise, the output makes no sense.
    """
    # If one of them is zero, then the other should be considered to be 0.
    # In this problem formulation, log0 = 0
    p = p.replace(0, np.nan)
    q = q.replace(0, np.nan)

    return (np.log2(p / q) * p).sum(axis=0)


def jsd(p, q):
    """Finds the per-column JSD betwen dataframes p and q

    Jensen-Shannon divergence of two probability distrubutions pandas
    dataframes, p and q. These distributions are usually created by running
    binify() on the dataframe.

    Parameters
    ----------
    p : pandas.DataFrame
        An nbins x features DataFrame.
    q : pandas.DataFrame
        An nbins x features DataFrame.

    Returns
    -------
    jsd : pandas.Series
        Jensen-Shannon divergence of each column with the same names between
        p and q
    """
    weight = 0.5
    m = weight * (p + q)

    result = weight * kld(p, m) + (1 - weight) * kld(q, m)
    return result


def entropy(binned, base=2):
    """Find the entropy of each column of a dataframe

    Parameters
    ----------
    binned : pandas.DataFrame
        A nbins x features DataFrame of probability distributions, where each
        column sums to 1
    base : numeric
        The log-base of the entropy. Default is 2, so the resulting entropy
        is in bits.

    Returns
    -------
    entropy : pandas.Seires
        Entropy values for each column of the dataframe.
    """
    return -((np.log(binned) / np.log(base)) * binned).sum(axis=0)


def binify_and_jsd(df1, df2, pair, bins):
    binned1 = binify(df1, bins=bins).dropna(how='all', axis=1)
    binned2 = binify(df2, bins=bins).dropna(how='all', axis=1)

    binned1, binned2 = binned1.align(binned2, axis=1, join='inner')

    series = np.sqrt(jsd(binned1, binned2))
    series.name = pair
    return series


def cross_phenotype_jsd(grouped, bins, n_iter=100):
    jsds = []

    seen = set([])

    for phenotype1, df1 in grouped:
        for phenotype2, df2 in grouped:
            pair = tuple(sorted([phenotype1, phenotype2]))
            print pair
            if pair in seen:
                continue
            seen.add(pair)

            if phenotype1 == phenotype2:
                seriess = []
                bs = cross_validation.Bootstrap(df1.shape[0], n_iter=n_iter,
                                                train_size=0.5)
                for i, (ind1, ind2) in enumerate(bs):
                    df1_subset = df1.iloc[ind1, :]
                    df2_subset = df2.iloc[ind2, :]
                    seriess.append(
                        binify_and_jsd(df1_subset, df2_subset, None, bins))
                series = pd.concat(seriess, axis=1, names=None).mean(axis=1)
                series.name = pair
                jsds.append(series)
            else:
                series = binify_and_jsd(df1, df2, pair, bins)
                jsds.append(series)
    return pd.concat(jsds, axis=1)

def jsd_df_to_2d(jsd_df):
    jsd_2d = jsd_df.mean().reset_index()
    jsd_2d = jsd_2d.rename(
        columns={'level_0': 'phenotype0', 'level_1': 'phenotype1', 0: 'jsd'})
    jsd_2d = jsd_2d.pivot(index='phenotype0', columns='phenotype1',
                          values='jsd')
    return jsd_2d + np.tril(jsd_2d.T, -1)