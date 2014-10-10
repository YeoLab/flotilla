FROM ubuntu

MAINTAINER Michael Lovci <michaeltlovci@gmail.com>

RUN apt-get update && apt-get install -q -y wget libglib2.0-0 libsm6 libx11-dev libxrender1 libfontconfig1 libxext6 gcc ftp build-essential zlibc zlib1g zlib1g-dev openssl libssl-dev mysql-server libmysqlclient-dev ghostscript zip emacs npm git curl libncurses5-dev default-jre default-jdk

ADD http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh /home/root/software/

WORKDIR /home/root/software

RUN ["chmod", "u+x", "Miniconda-3.7.0-Linux-x86_64.sh"]
RUN ./Miniconda-3.7.0-Linux-x86_64.sh -b -p /usr/local/miniconda -f

ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/miniconda/bin
RUN conda install --yes cython numpy scipy scikit-learn pip pandas ipython jinja2 tornado pyzmq matplotlib pyqt
RUN pip install patsy && pip install seaborn brewer2mpl networkx semantic_version statsmodels htseq misopy pybedtools gffutils cutadapt pysam
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
MOUNT /root/ipython
#user should use -v option to mount ~/flotilla_packages here
MOUNT /root/flotilla_packages

EXPOSE 8888
ENTRYPOINT run_notebook.sh
