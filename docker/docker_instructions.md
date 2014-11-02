Here are instructions to get an active docker image. These instructions have not been tested on Windows.

Note: On Mac OS X and Windows you will need to start docker through the “boot2docker” application before you can use docker.

  1. Install docker ( ≥ version 1.3) according to the [instructions appropriate for your system](https://docs.docker.com/installation/#installation).<br>
  2. Then, obtain the flotilla image with:


    sudo docker pull mlovci/flotilla

Then start flotilla with:

    sudo docker run -v ${HOME}/flotilla_projects:/root/flotilla_projects \ #mount flotilla resouces from the host 
                    -v ${HOME}/flotilla_notebooks:/root/home/ipython \ # mount the notebook directory to a local directory
                    -it -P -p 8888:8888 \ #run interactively and map ports to host
                    mlovci/flotilla:latest

    ( or mlovci/flotilla:dev )

Or, magic web-start (I think this works, at least it does on local tests):

    curl https://raw.githubusercontent.com/YeoLab/flotilla/dev/docker/start_docker.py | python


This command will cross-mount directories on your computer to a location available for reading 
and writing inside the virtual machine. Any analysis you perform will be output to ${HOME}/flotilla_notebooks 
and the data flotilla requires will be saved in ${HOME}/flotilla_projects.


