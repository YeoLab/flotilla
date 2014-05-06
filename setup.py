from setuptools import setup
from setuptools import find_packages

setup(
    name='flotilla',
    packages=find_packages(),
    url='',
    license='',
    author='lovci',
    author_email='mlovci@ucsd.edu',
    description='',
    include_package_data=True,
    package_dir={'flotilla': 'src/_cargo_commonObjects/cargo_data/'}
    package_data={
        'flotilla': ['*/*txt.gz',
        '*/gene_lists/*']
    },
    entry_points={
    'console_scripts': ['metrics_runner.py = flotilla.src.carrier',
                        'serve_flotilla_notebook = flotilla.src._barge_utils:serve_ipython']},
    install_requires=['setuptools',
                      'numpy >= 1.6.1 ',
                      'scipy >= 0.13.0',
                      'matplotlib >= 1.3.1',
                      'scikit-learn >= 0.13.0',
                      'gspread',
                      'brewer2mpl',
                      'pymongo >= 2.7',
                      'ipython >= 2.0.0',
                      'husl',
                      'seaborn >= 0.3.1',
                      'patsy >= 0.2.1',                      
                      'statsmodels >= 0.5.0',
                      'pandas >= 0.13.1',
                      'networkx',
                      'tornado >= 3.2.1',
                      #'dcor_cpy' #needs to be built with extutils
    ],
    version = "0.0.1"
)
