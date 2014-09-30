from setuptools import setup
from setuptools import find_packages

from flotilla import __version__


version = __version__

setup(
    name='flotilla',
    packages=find_packages(),
    url='http://github.com/YeoLab/flotilla',
    license='',
    author='mlovci,olgabot',
    author_email='obotvinn@ucsd.edu',
    description='Embark on a journey of single-cell data exploration.',
    install_requires=open('requirements.txt').readlines(),
    version=version,
    classifiers=['License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Topic :: Scientific/Engineering :: Visualization',
                 'Topic :: Scientific/Engineering :: Medical Science Apps.,'
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Multimedia :: Graphics',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows'
    ]
)
