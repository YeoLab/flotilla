FROM mlovci/anaconda_python

MAINTAINER Michael Lovci <michaeltlovci@gmail.com>


RUN adduser --disabled-password --gecos '' --home=/home/flotilla flotilla

WORKDIR /usr/bin

ADD https://raw.githubusercontent.com/YeoLab/flotilla/dev/docker/scripts/run_notebook.sh /usr/bin/run_notebook.sh
RUN chmod 755 run_notebook.sh

RUN apt-get install -y r-base
RUN pip install --upgrade rpy2

WORKDIR /home/root
ADD https://raw.githubusercontent.com/YeoLab/flotilla/dev/docker/scripts/monocole_deps_installer.R /home/root/monocle_deps_installer.R
RUN Rscript /home/root/monocle_deps_installer.R

ADD http://monocle-bio.sourceforge.net/downloads/monocle_0.99.0.tar.gz /home/root/monocle_0.99.0.tar.gz
ADD http://monocle-bio.sourceforge.net/downloads/HSMMSingleCell_0.99.0.tar.gz /home/root/HSMMSingleCell_0.99.0.tar.gz

RUN R CMD INSTALL HSMMSingleCell_0.99.0.tar.gz
RUN R CMD INSTALL monocle_0.99.0.tar.gz

WORKDIR /home/root/ipython


#this part needs a solution to https://github.com/docker/docker/issues/5189 but it would be preferred if the notebook were run as a flotilla user
#USER flotilla
#ENV HOME /home/flotilla
#VOLUME /home/flotilla/ipython
#VOLUME /home/flotilla/flotilla_packages

RUN pip install -e git://github.com/YeoLab/flotilla.git@dev#egg=flotilla

ENV HOME /root
#user should use -v option to mount a host directory here
VOLUME /root/ipython
#user should use -v option to mount ~/flotilla_packages here
VOLUME /root/flotilla_packages

EXPOSE 8888
ENTRYPOINT run_notebook.sh
