import numpy as np
import pandas as pd


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
    """Makes a histogram of each row the provided binsize

    Parameters
    ----------
    dataset : pandas.DataFrame
        A samples x features dataframe. Each feature will be binned into the
        provided bins
    bins : iterable
        Bins you would like to use for this dataset. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1).
        nbins = len(bins) - 1

    Returns
    -------
    binned : pandas.DataFrame
        An nbins x features DataFrame of each column binned across rows
    """
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
        A nbins x features DataFrame
    base : numeric
        The log-base of the entropy. Default is 2, so the resulting entropy
        is in bits.

    Returns
    -------
    entropy : pandas.Seires
        Entropy values for each column of the dataframe.
    """
    return -((np.log(binned) / np.log(base)) * binned).sum(axis=0)
