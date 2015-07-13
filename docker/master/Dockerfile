FROM mlovci/anaconda_python

MAINTAINER Michael Lovci <michaeltlovci@gmail.com>

RUN pip install -e git://github.com/YeoLab/flotilla.git@master#egg=flotilla

RUN adduser --disabled-password --gecos '' --home=/home/flotilla flotilla

WORKDIR /usr/bin

ADD https://gist.githubusercontent.com/mlovci/74c96dda49680419bcca/raw/15029fffa38585360502eee4d11a2a5ec20f372f/run_notebook.sh /usr/bin/run_notebook.sh
RUN chmod 755 run_notebook.sh

WORKDIR /home/root/ipython


#this part needs a solution to https://github.com/docker/docker/issues/5189 but it would be preferred if the notebook were run as a flotilla user
#USER flotilla
#ENV HOME /home/flotilla
#MOUNT /home/flotilla/ipython
#MOUNT /home/flotilla/flotilla_packages

ENV HOME /root
#user should use -v option to mount a host directory here
VOLUME /root/ipython
#user should use -v option to mount ~/flotilla_packages here
VOLUME /root/flotilla_packages

EXPOSE 8888
ENTRYPOINT run_notebook.sh
