import numpy as np
import pandas as pd


def kld(P, Q):
    """Kullback-Leiber divergence of two probability distributions pandas
    dataframes, P and Q

    Parameters
    ----------


    Returns
    -------


    Raises
    ------
    """
    # If one of them is zero, then the other should be considered to be 0
    P = P.replace(0, np.nan)
    Q = Q.replace(0, np.nan)

    return (np.log2(P / Q) * P).sum(axis=0)


def jsd(P, Q):
    """Finds the per-column JSD betwen dataframes P and Q

    Jensen-Shannon divergence of two probability distrubutions pandas
    dataframes, P and Q. These distributions are usually created by running
    binify() on the dataframe.

    Parameters
    ----------


    Returns
    -------


    Raises
    ------
    """
    weight = 0.5
    M = weight * (P + Q)

    result = weight * kld(P, M) + (1 - weight) * kld(Q, M)
    return result


def entropy(binned, base=2):
    """
    Given a binned dataframe created by 'binify', find the entropy of each
    row (index)
    """
    return -((np.log2(binned) / np.log2(base)) * binned).sum(axis=0)


def binify(df, bins):
    """Makes a histogram of each row the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        A samples x features dataframe. Each feature will be binned into the
        provided bins
    bins : iterable
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1)

    Returns
    -------
    binned : pandas.DataFrame
        A len(bins)-1 x features DataFrame of each feature binned across
        samples
    """
    binned = df.apply(lambda x: pd.Series(np.histogram(x, bins=bins)[0]))
    binned.index = make_bin_range_strings(bins)

    # Normalize so each column sums to 1
    binned = binned / binned.sum().astype(float)
    return binned


def make_bin_range_strings(bins):
    return ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]