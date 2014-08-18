from setuptools import setup
from setuptools import find_packages

setup(
    name='flotilla',
    packages=find_packages(),
    url='http://github.com/YeoLab/flotilla',
    license='',
    author='mlovci,olgabot',
    author_email='mlovci@ucsd.edu',
    description='Embark on a journey of data exploration.',
    install_requires=open('requirements.txt').readlines(),
    version="0.1.0"
)
