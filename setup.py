
from setuptools import setup
from setuptools import find_packages

setup(
    name = 'flotilla',
    version = '',
    packages = find_packages(),
    url = '',
    license = '',
    author = 'lovci',
    author_email = 'mlovci@ucsd.edu',
    description = '',
    include_package_data=True,
    package_data = {
        'flotilla' : ['data/*', 'project/data/*']
        },
    entry_points = {'console_scripts':['metrics_runner.py = flotilla.src.carrier',
                                       'flotilla_notebook = flotilla.src.barge:serve_ipython']},
    install_requires = ['setuptools',
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
                        'statsmodels >= 0.5.0',
                        'patsy >= 0.2.1',
                        'networkx',
                        'dcor_cpy' #needs to be built with extutils
                        ],

)
