"""
Installation setup
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from setuptools import setup
from setuptools import find_packages

import flotilla

version = flotilla.__version__

setup(
    name='flotilla',
    packages=find_packages(),
    url='http://github.com/YeoLab/flotilla',
    license='3-Clause BSD',
    author='olgabot',
    author_email='obotvinn@ucsd.edu',
    description='Embark on a journey of single-cell data exploration.',
    # These are all in a specific order! If you add another package as
    # a depencency, please make sure that its dependencies come before it
    # (above) in the list. E.g. "numpy" and "scipy" must precede "matplotlib"
    install_requires=["setuptools",
                      "numpy >= 1.8.0",
                      "scipy >= 0.14",
                      "matplotlib >= 1.3.1",
                      "scikit-learn >= 0.13.0",
                      "brewer2mpl",
                      "pymongo >= 2.7",
                      "ipython >= 2.0.0",
                      "husl",
                      "patsy >= 0.2.1",
                      "pandas >= 0.13.1",
                      "statsmodels >= 0.5.0",
                      "seaborn >= 0.3",
                      "networkx",
                      "tornado >= 3.2.1",
                      "pyzmq",
                      # "dcor_cpy' #needs to be built with extutils",
                      "six",
                      "pytest-cov",
                      "python-coveralls",
                      "jinja2",
                      # "fastcluster",
                      "semantic_version",
                      "joblib >= 0.8.4"],
    version=version,
    classifiers=['License :: OSI Approved :: BSD License',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Topic :: Scientific/Engineering :: Visualization',
                 'Topic :: Scientific/Engineering :: Medical Science Apps.',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Multimedia :: Graphics',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows']
)
