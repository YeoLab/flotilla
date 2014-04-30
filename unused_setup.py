
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
    entry_points = {'console_scripts':['metrics_runner.py = flotilla.src.carrier']}
)
