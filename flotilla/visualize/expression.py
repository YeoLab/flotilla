__author__ = 'olga'

import matplotlib.pyplot as plt
import numpy as np

from .color import red, green, blue
from ..compute.expression import TwoWayGeneComparisonLocal

class TwoWayScatterViz(TwoWayGeneComparisonLocal):

    def __call__(self, **kwargs):
        self.plot(**kwargs)

    def plot(self, ax=None):

        co = [] #colors container
        for label, (pVal, logratio, isSig) in self.result_.get(["pValue", "log2Ratio", "isSig"]).iterrows():
            if (pVal < self.pCut) and isSig:
                if logratio > 0:
                    co.append(red)
                elif logratio < 0:
                    co.append(green)
                else:
                    raise Exception
            else:
                co.append(blue)

        if ax == None:
            ax = plt.gca()

        ax.set_aspect('equal')
        vmin = np.min(np.c_[self.sample1, self.sample2])
        ax.scatter(self.sample1, self.sample2, c=co, alpha=0.7, edgecolor='none')
        ax.set_xlabel("%s %s" % (self.sampleNames[0], self.dtype))
        ax.set_ylabel("%s %s" % (self.sampleNames[1], self.dtype))
        ax.set_yscale('log', basey=2)
        ax.set_xscale('log', basex=2)
        ax.set_xlim(xmin=max(vmin, 0.1))
        ax.set_ylim(ymin=max(vmin, 0.1))
        if ax == None:
            plt.tight_layout()