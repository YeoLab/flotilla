import numpy as np
import pandas as pd


def kld(P, Q):
    """
    Kullback-Leiber divergence of two probability distributions pandas
    dataframes, P and Q
    """
    return (np.log2(P / Q) * P).sum(axis=1)


def jsd(P, Q):
    """Finds the per-row JSD betwen dataframes P and Q

    Jensen-Shannon divergence of two probability distrubutions pandas
    dataframes, P and Q. These distributions are usually created by running
    binify() on the dataframe.
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
    return -((np.log2(binned) / np.log2(base)) * binned).sum(axis=1)


def binify(df, bins):
    """Makes a histogram of each row the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe whose rows you'd like to binify.
    bins : iterable
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1)

    Returns
    -------
    binned : pandas.DataFrame

    Raises
    ------


    """
    ncol = len(bins) - 1
    nrow = df.shape[0]
    binned = np.zeros((nrow, ncol))

    # TODO.md: make sure this works for numpy matrices
    for i, (name, row) in enumerate(df.iterrows()):
        binned[i, :] = np.histogram(row, bins=bins, normed=True)[0]

    columns = ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]
    binned = pd.DataFrame(binned, index=df.index, columns=columns)
    return binned