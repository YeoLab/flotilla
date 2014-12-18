OS X Installation instructions
==============================

The following steps have been tested on a clean install of Mavericks
10.9.4.

All others must fend for themselves to install ``matplotlib``, ``scipy`` and
their third-party dependencies.


To install, first install the Anaconda_ Python Distribution, which comes
pre-packaged with a bunch of the scientific packages we use all the time,
pre-installed.

Create a flotilla sandbox
-------------------------

We recommend creating a "sandbox" where you can install any and all packages
without disturbing the rest of the Python distribution. This way, your
flotilla environment is completely insulated from everything else, just in
case something goes wrong with You can do this with

.. code::

    conda create --yes flotilla_env pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml

You've now just created a "virtual environment" called ``flotilla_env`` (the first
argument). Now activate that environment with,

.. code::

    source activate flotilla_env

Now at the beginning of your terminal prompt, you should see:

.. code::

    (flotilla_env)

Which indicates that you are now in the ``flotilla_env`` virtual environment. Now
that you're in the environment, follow along with the non-sandbox
installation instructions.

Install and update all packages in your environment
---------------------------------------------------

To make sure you have the latest packages, on the command line in your
terminal, enter this command:

.. code::

    conda update --yes pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml

Not all packages are available using ``conda``, so we'll install the rest using
``pip``, which is a Python package installer and installs from PyPI_, the
Python Package Index.

.. code::

    pip install seaborn fastcluster brewer2mpl pymongo gspread husl semantic_version joblib

Next, to install the latest release of ``flotilla``, do

.. code::

    pip install flotilla

If you want the bleeding-edge master version because you're a developer or a
super-user, (we work really hard to make sure it's always working but could be
buggy!), then install the ``git`` master with,

.. code::

    pip install git+git://github.com/yeolab/flotilla.git

Old OSX Installation instructions
=================================

*This part only needs to be done once*

-  Install Anaconda_
-  Install XCode_ (This can take an hour)
-  Install homebrew_, via

.. code::

   ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"

-  Install freetype:

.. code::

   brew install freetype

-  Install heavy packages (this can take an hour or more)

.. code::

    conda install --yes pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml`

-  Create a virtual environment

.. code::

   conda create --yes -n flotilla_env pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml

-  Switch to virtual environment

.. code::

   source activate flotilla_env

-  Install flotilla and its dependencies (this can take a few minutes):

.. code::

    pip install git+https://github.com/YeoLab/flotilla.git

-  Create a scratch space for your work

.. code::

   mkdir ~/flotilla_scratch

-  Make a place to store flotilla projects

.. code::

   mkdir ~/flotilla_projects

-  Go back to the real world

.. code::
   ``source deactivate``

-  switch to virtual environment

   ``source activate flotilla_env``

-  start an ipython notebook:

   ``ipython notebook --notebook-dir=~/flotilla_scratch``

-  create a new notebook by clicking ``New Notebook``
-  rename your notebook from "Untitled" to something more informative by
   clicking the title panel.

.. include:: docker_instructions.rst

.. _Anaconda: http://continuum.io/downloads
.. _XCode: https://itunes.apple.com/us/app/xcode/id497799835?mt=12
.. _homebrew: http://brew.sh/
.. _PyPI: https://pypi.python.org/