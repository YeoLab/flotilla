from setuptools import setup
from setuptools import find_packages

setup(
    name = 'barbones_project',
    package_data={'barbones_project': ['study_data/*']},
    packages = find_packages(),
    url = '',
    license = '',
    author = 'lovci',
    author_email = '',
    description = '',
    version = '0.0.1'
)
#TODO: fix: if data frames are large (>100MB), they can't go to github.
#fix this by specifying source files to download... from our server?
