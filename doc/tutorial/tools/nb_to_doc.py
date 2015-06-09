#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.

From seaborn's tutorial building
"""
import os
import sys


def convert_nb(nbname):

    # os.system("runipy --o %s --matplotlib --quiet" % nbname)
    os.system("ipython nbconvert --execute --ExecutePreprocessor.timeout=240"
              " --template tools/rst_custom --to rst "
              "--Exporter.preprocessors=\"['IPython.nbconvert.preprocessors.ExtractOutputPreprocessor']\"  %s" % nbname)
    os.system("tools/nbstripout %s" % nbname)


if __name__ == "__main__":

    for nbname in sys.argv[1:]:
        convert_nb(nbname)