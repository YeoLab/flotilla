Docker Installation Instructions
================================

[Docker](https://www.docker.com/whatisdocker/) is the preferred method to obtain the most up-to-date
version of `flotilla`. Every change we make to the source code triggers a new build of a virtual
 machine that contains flotilla and all its dependencies.
 
Please follow instructions [here](docker/docker_instructions.md) to get, install, and run the `flotilla` image.



OS X Installation instructions
==============================

The following steps have been tested on a clean install of Mavericks 10.9.4.

All others must fend for themselves to install matplotlib, scipy and their third-party dependencies.

 *This part only needs to be done once*

 * [Install anaconda](https://store.continuum.io/cshop/anaconda/)
 * [Install Xcode (this can take an hour)](https://itunes.apple.com/us/app/xcode/id497799835?mt=12)
 * Open Xcode and agree to terms and services (it is very important to read them thoroughly)
 * Install [homebrew](http://brew.sh/)


    `ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"`


 * Install freetype:


    `brew install freetype`


 * Install heavy packages (this can take an hour or more)

```
conda install pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml`
```

 * Create a virtual environment
```
conda create -n flotilla_env pip numpy scipy cython matplotlib nose six scikit-learn ipython networkx pandas tornado statsmodels setuptools pytest pyzmq jinja2 pyyaml`
```

 * Switch to virtual environment


    `source activate flotilla_env`


 * Install flotilla and its dependencies (this can take a few minutes):


    `pip install git+https://github.com/YeoLab/flotilla.git`


 * Create a scratch space for your work


    `mkdir ~/flotilla_scratch`


 * Make a place to store flotilla projects


    `mkdir ~/flotilla_projects`


 * Go back to the real world

    `source deactivate`
    
    
 
 * switch to virtual environment


    `source activate flotilla_env`


 * start an ipython notebook:


    `ipython notebook --notebook-dir=~/flotilla_scratch`


 * create a new notebook by clicking `New Notebook`
 * rename your notebook from "Untitled" to something more informative by clicking the title panel.