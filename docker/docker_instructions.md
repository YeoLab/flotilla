Here are instructions to get an active docker image.

Note: On Mac OS X and Windows you will need to start docker through the “boot2docker” application before you can use docker.

  1. Install docker ( ≥ version 1.3) according to the [instructions appropriate for your system](https://docs.docker.com/installation/#installation).<br>
  2. Then, obtain the flotilla image with:

docker pull mlovci/flotilla

Then start flotilla with:

    docker run -v ${HOME}/flotilla_projects:/root/flotilla_projects \
               -v ${HOME}/flotilla_notebooks:/root/home/ipython \
               -it -P -p 8888:8888 \
               mlovci/flotilla:latest

    ( or mlovci/flotilla:dev )


This command will cross-mount directories on your computer to a location available for reading<br>
 and writing inside the virtual machine. Any analysis you perform will be output to ${HOME}/flotilla_notebooks<br>
  and the data flotilla requires will be saved in ${HOME}/flotilla_projects.


