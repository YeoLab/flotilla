from __future__ import division
import itertools
import math
import sys

import numpy as np
from scipy import stats
import pandas as pd


def benjamini_hochberg(p_values, fdr=0.1):
    """Benjamini-Hochberg correction for multiple hypothesis testing

    From: http://udel.edu/~mcdonald/statmultcomp.html
    One good technique for controlling the false discovery rate was briefly
    mentioned by Simes (1986) and developed in detail by Benjamini and Hochberg
    (1995). Put the individual P-values in order, from smallest to largest.
    The smallest P-value has a rank of i=1, the next has i=2, etc. Then
    compare each individual P-value to (i/m)Q, where m is the total number of
    test and Q is the chosen false discovery rate. The largest P-value that
    has P<(i/m)Q is significant, and all P-values smaller than it are also
    significant.

    Parameters
    ----------
    p_values : list
        List of p-values
    fdr : float, optional
        Desired false-discovery rate cutoff

    Returns
    -------
    sigs : numpy.array
        Boolean array of whether or not the provided p-values are significant
        given the FDR cutoff
    """
    nComps = len(p_values) + 0.0
    pSorter = np.argsort(p_values)
    pRank = np.argsort(np.argsort(p_values)) + 1
    BHcalc = (pRank / nComps) * fdr
    sigs = np.ndarray(shape=(nComps, ), dtype='bool')
    issig = True
    for (p, b, r) in itertools.izip(p_values[pSorter], BHcalc[pSorter],
                                    pSorter):
        if p > b:
            issig = False
        sigs[r] = issig
    return sigs


class TwoWayGeneComparisonLocal(object):
    """Compare gene expression for two samples
    """
    def __init__(self, sample1_name, sample2_name, df, p_value_cutoff=0.001,
                 local_fraction=0.1, bonferroni=True, fdr=None,
                 dtype="RPKM"):
        """

        Plots a scatter-plot of sample1 vs sample2, taken from df.
        Calculates differentially expressed genes with a Z-test from
        the closest (local_fraction * 100)%  points. Stores result from
        statistical calculations in self.result_

        Parameters
        ----------
        sample1_name : str
            Name of the first (control) sample. Must be a row name (index) in
            df. Plotted on the x-axis.
        sample2_name : str
            Name of the second (treatment) sample. Must be a row name (index)
            in df. Plotted on the y-axis.
        df : pandas.DataFrame
            A samples (rows) x features (columns) pandas DataFrame of
            expression values
        p_value_cutoff : float, optional
            Cutoff for the p-values. Default 0.001.
        local_fraction : float, optional
            What fraction of genes to use for *local* z-score calculation.
            Default 0.1
        bonferonni : bool, optional
            Whether or not to use the Bonferonni correction on p-values
        fdr : ???, optional
            benjamini-hochberg FDR filtering - check result, proceed with
            caution. sometimes breaks :(
        dtype : str, optional
            Data type
        """

        sample1 = df.ix[sample1_name]
        sample2 = df.ix[sample2_name]

        self.sample_names = (sample1.name, sample2.name)

        sample1 = sample1.replace(0, np.nan).dropna()
        sample2 = sample2.replace(0, np.nan).dropna()

        sample1, sample2 = sample1.align(sample2, join='inner')

        self.sample1 = sample1
        self.sample2 = sample2
        labels = sample1.index

        self.n_genes = len(labels)
        if bonferroni:
            correction = self.n_genes
        else:
            correction = 1

        local_count = int(math.ceil(self.n_genes * local_fraction))
        self.p_value_cutoff = p_value_cutoff
        self.upregulated_genes = set()
        self.downregulated_genes = set()
        self.expressed_genes = set([labels[i] for i, t in enumerate(
            np.any(np.c_[sample1, sample2] > 1, axis=1)) if t])
        self.log2_ratio = np.log2(sample2 / sample1)
        self.average_expression = (sample2 + sample1) / 2.
        self.ranks = np.argsort(np.argsort(self.average_expression))
        self.p_values = pd.Series(index=labels)
        self.local_mean = pd.Series(index=labels)
        self.local_std = pd.Series(index=labels)
        self.local_z = pd.Series(index=labels)
        self.dtype = dtype

        for g, r in itertools.izip(self.ranks.index, self.ranks):
            if r < local_count:
                start = 0
                stop = local_count

            elif r > self.n_genes - local_count:
                start = self.n_genes - local_count
                stop = self.n_genes

            else:
                start = r - int(math.floor(local_count / 2.))
                stop = r + int(math.ceil(local_count / 2.))

            local_genes = self.ranks[self.ranks.between(start, stop)].index
            self.local_mean.ix[g] = np.mean(self.log2_ratio.ix[local_genes])
            self.local_std.ix[g] = np.std(self.log2_ratio.ix[local_genes])
            self.p_values.ix[g] = stats.norm.pdf(self.log2_ratio.ix[g],
                                                 self.local_mean.ix[g],
                                                 self.local_std.ix[
                                                     g]) * correction
            self.local_z.ix[g] = (self.log2_ratio.ix[g] - self.local_mean.ix[
                g]) / self.local_std.ix[g]

        data = pd.DataFrame(index=labels)
        data["rank"] = self.ranks
        data["log2_ratio"] = self.log2_ratio
        data["local_mean"] = self.local_mean
        data["local_std"] = self.local_std
        data["pValue"] = self.p_values

        if fdr is None:
            data["isSig"] = self.p_values < p_value_cutoff
        else:
            data["isSig"] = benjamini_hochberg(self.p_values, fdr=fdr)

        data["meanExpression"] = self.average_expression
        data["local_z"] = self.local_z
        data[self.sample_names[0]] = sample1
        data[self.sample_names[1]] = sample2

        self.result_ = data

        for label, (pVal, logratio, isSig) in data.get(
                ["pValue", "log2_ratio", "isSig"]).iterrows():
            if (pVal < p_value_cutoff) and isSig:
                if logratio > 0:
                    self.upregulated_genes.add(label)
                elif logratio < 0:
                    self.downregulated_genes.add(label)
                else:
                    raise ValueError

    def gstats(self):
        """Write general statistics of the two-way comparison to standard output
        """
        sys.stdout.write(
            "I used a p-value cutoff of {:.2e}\n".format(self.p_value_cutoff))
        sys.stdout.write("\tThere are {} up-regulated genes in {} vs {}\n"
                         .format(len(self.upregulated_genes),
                                 self.sample_names[1],
                                 self.sample_names[0]))
        sys.stdout.write("\tThere are {} down-regulated genes in %s vs %s"
                         .format(len(self.downregulated_genes),
                                 self.sample_names[1],
                                 self.sample_names[0]))
        sys.stdout.write("There are {} expressed genes in both {} and {}"
                         .format(len(self.expressed_genes),
                                 *self.sample_names))


def differential_expression(data, groupby):
    """Calculate probability that a feature's values are skewed towards a group

    Uses a Mann-Whitney U test when the number of groups is equal to 2, and
    a Kruskal-Wallis test when the number of groups is greater than 2. If
    there are fewer than 2 groups, raises a ValueError.

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_samples, n_features) matrix
    groupby : pandas.Series
        A (n_samples,) Series describing the group membership of the samples

    Returns
    -------
    de_results : pandas.DataFrame
        A (n_features, 4) dataframe with the columns

    Raises
    ------
    ValuError
        If the number of groups in the groupby is fewer than 2.
    """

    n_groups = len(groupby.groupby(groupby).size())
    if n_groups == 2:
        statistical_test = stats.mannwhitneyu
    elif n_groups > 2:
        statistical_test = stats.kruskal
    else:
        raise ValueError('Must have at least two groups to calculate '
                         'differential expression')
    de_results = dict((col,
        pd.Series(statistical_test(*[s for diagnosis, s
                                     in series.groupby(groupby)]),
            index=['U_statistic', 'p_value']))
     for col, series in data.iteritems())
    de_results = pd.DataFrame.from_records(de_results).T
    de_results['bonferonni_p_value'] = de_results.p_value*de_results.shape[0]
    de_results = de_results.sort('bonferonni_p_value')
    df = de_results.reset_index()
    df = df.rename(columns={'index': 'gene_id'})
    return df
