
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
    package_data = {
        'flotilla' : ['data/*.gff', 'data/regions/*.bed']
        },
)
