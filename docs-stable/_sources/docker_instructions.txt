Docker Instructions
===================


`Docker <https://www.docker.com/whatisdocker/>`__ is the preferred
method to obtain the most up-to-date version of ``flotilla``. Every
change we make to the source code triggers a new build of a virtual
machine that contains flotilla and all its dependencies.


Here are instructions to get an active docker image. These instructions
have not been tested on Windows or Linux.

Note: On Mac OS X and Windows you will need to start docker through the
“boot2docker” application before you can use docker.

1. Install docker ( ≥ version 1.3) according to the `instructions
   appropriate for your
   system <https://docs.docker.com/installation/#installation>`__.
2. Then start flotilla on the command line (OS X ``Terminal``
   application):

   curl
   https://raw.githubusercontent.com/YeoLab/flotilla/dev/docker/start\_docker.py
   \| python

After the ipython notebook interface opens, test the installation with
our test dataset by running the following commands in a new notebook:

::

    import flotilla
    study = flotilla.embark("http://sauron.ucsd.edu/flotilla_projects/neural_diff_chr22/datapackage.json")
    study.interactive_pca()

Thanks for using flotilla!
