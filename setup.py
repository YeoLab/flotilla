from setuptools import setup
from setuptools import find_packages

setup(
    name='flotilla',
    packages=find_packages(),
    url='http://github.com/YeoLab/flotilla',
    license='',
    author='mlovci,olgabot',
    author_email='mlovci@ucsd.edu',
    description='',
    include_package_data=True,
    #package_dir={'flotilla': 'flotilla'},
    package_data={
        'flotilla': ['cargo/cargo_data/*/*txt.gz',
        'cargo/cargo_data/*/gene_lists/*'
        ]
    },
    entry_points={
    'console_scripts': ['metrics_runner.py = flotilla.carrier',
                        'serve_flotilla_notebook = flotilla._barge_utils:serve_ipython']},
    install_requires=open('requirements.txt').readlines(),
    version = "0.0.4"
)
