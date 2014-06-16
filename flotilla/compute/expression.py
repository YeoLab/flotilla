__author__ = 'olga'

import numpy as np
import itertools
from scipy import stats
import pandas as pd
import math
from __future__ import division

def benjamini_hochberg(p_values, fdr=0.1):
    """ benjamini-hochberg correction for MHT
        p_values is a list of p_values
        fdr is the desired false-discovery rate

        from: http://udel.edu/~mcdonald/statmultcomp.html
        "One good technique for controlling the false discovery rate was briefly
        mentioned by Simes (1986) and developed in detail by Benjamini and Hochberg (1995).
        Put the individual P-values in order, from smallest to largest. The smallest
        P-value has a rank of i=1, the next has i=2, etc. Then compare each individual
        P-value to (i/m)Q, where m is the total number of test and Q is the chosen false
        discovery rate. The largest P-value that has P<(i/m)Q is significant,
        and all P-values smaller than it are also significant."

        """
    ranks = np.argsort(np.argsort(p_values))

    nComps = len(p_values) + 0.0
    pSorter = np.argsort(p_values)
    pRank = np.argsort(np.argsort(p_values))+1
    BHcalc = (pRank / nComps) * fdr
    sigs = np.ndarray(shape=(nComps, ), dtype='bool')
    issig = True
    for (p, b, r) in itertools.izip(p_values[pSorter], BHcalc[pSorter], pSorter):
        if p > b:
            issig = False
        sigs[r] = issig
    return sigs


class TwoWayGeneComparisonLocal(object):

    def __init__(self, sample1_name, sample2_name, df, pCut = 0.001,
                 local_fraction = 0.1, bonferroni = True, FDR=None,
                 dtype="RPKM"):
        """ Run a two-sample RPKM experiment.
            Give control sample first, it will go on the x-axis
            df is a pandas dataframe with features (genes) on columns and samples on rows
            sample1 and sample2 are the names of rows in df (sample IDs)
            pCut - P value cutoff
            local_fraction - by default the closest 10% of genes are used for local z-score calculation
            bonferroni - p-values are adjusted for MHT with bonferroni correction
            BH - benjamini-hochberg FDR filtering - check result, proceed with caution. sometimes breaks :(
        """

        sample1 = df.ix[sample1_name]
        sample2 = df.ix[sample2_name]

        sampleNames = (sample1.name, sample2.name)
        self.sampleNames = sampleNames

        sample1 = sample1.replace(0, np.nan).dropna()
        sample2 = sample2.replace(0, np.nan).dropna()

        sample1, sample2 = sample1.align(sample2, join='inner')

        self.sample1 = sample1
        self.sample2 = sample2
        labels = sample1.index

        self.nGenes = len(labels)
        if bonferroni:
            correction = self.nGenes
        else:
            correction = 1

        localCount = int(math.ceil(self.nGenes * local_fraction))
        self.pCut = pCut
        self.upGenes = set()
        self.dnGenes = set()
        self.expressedGenes = set([labels[i] for i, t in enumerate(np.any(np.c_[sample1, sample2] > 1, axis=1)) if t])
        self.log2Ratio = np.log2(sample2 / sample1)
        self.average_expression = (sample2 + sample1)/2.
        self.ranks = np.argsort(np.argsort(self.average_expression))
        self.pValues = pd.Series(index = labels)
        self.localMean = pd.Series(index = labels)
        self.localStd = pd.Series(index = labels)
        self.localZ = pd.Series(index = labels)
        self.dtype=dtype

        for g, r in itertools.izip(self.ranks.index, self.ranks):
            if r < localCount:
                start = 0
                stop = localCount

            elif r > self.nGenes - localCount:
                start = self.nGenes - localCount
                stop = self.nGenes

            else:
                start = r - int(math.floor(localCount/2.))
                stop = r + int(math.ceil(localCount/2.))

            localGenes = self.ranks[self.ranks.between(start, stop)].index
            self.localMean.ix[g] = np.mean(self.log2Ratio.ix[localGenes])
            self.localStd.ix[g] = np.std(self.log2Ratio.ix[localGenes])
            self.pValues.ix[g] = stats.norm.pdf(self.log2Ratio.ix[g],
                                                self.localMean.ix[g],
                                                self.localStd.ix[g]) * correction
            self.localZ.ix[g] = (self.log2Ratio.ix[g]- self.localMean.ix[g])/self.localStd.ix[g]

        data = pd.DataFrame(index = labels)
        data["rank"] = self.ranks
        data["log2Ratio"] = self.log2Ratio
        data["localMean"] = self.localMean
        data["localStd"] = self.localStd
        data["pValue"] = self.pValues

        if FDR == None:
            data["isSig"] = self.pValues < pCut
        else:
            data["isSig"] = benjamini_hochberg(self.pValues, fdr=FDR)

        data["meanExpression"] = self.average_expression
        data["localZ"] = self.localZ
        data[sampleNames[0]] = sample1
        data[sampleNames[1]] = sample2

        self.result_ = data

        for label, (pVal, logratio, isSig) in data.get(["pValue", "log2Ratio", "isSig"]).iterrows():
            if (pVal < pCut) and isSig:
                if logratio > 0:
                    self.upGenes.add(label)
                elif logratio < 0:
                   self.dnGenes.add(label)
                else:
                    raise ValueError

    def gstats(self):
        print "I used a p-value cutoff of %e" %self.pCut
        print "There are", len(self.upGenes), "up-regulated genes in %s vs %s" %(self.sampleNames[1],
                                                                                 self.sampleNames[0])
        print "There are", len(self.dnGenes), "down-regulated genes in %s vs %s" %(self.sampleNames[1],
                                                                                   self.sampleNames[0])
        print "There are", len(self.expressedGenes), "expressed genes in both %s and %s" %self.sampleNames