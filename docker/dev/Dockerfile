FROM mlovci/anaconda_python

MAINTAINER Michael Lovci <michaeltlovci@gmail.com>


RUN adduser --disabled-password --gecos '' --home=/home/flotilla flotilla

WORKDIR /usr/bin

ADD https://gist.githubusercontent.com/mlovci/74c96dda49680419bcca/raw/15029fffa38585360502eee4d11a2a5ec20f372f/run_notebook.sh /usr/bin/run_notebook.sh
RUN chmod 755 run_notebook.sh

RUN apt-get install -y r-base
RUN pip install --upgrade rpy2

WORKDIR /home/root
ADD https://gist.githubusercontent.com/mlovci/d9f9911741505d76c7be/raw/9a1b6d28d4cb3022f1fad8e71093eb1acf67bc7c/monocle_deps_installer.R /home/root/monocle_deps_installer.R
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
