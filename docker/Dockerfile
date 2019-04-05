FROM nvidia/cuda:9.1-base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
        python3-pip \
 && add-apt-repository -y ppa:jonathonf/python-3.6 \
 && apt-get update \
 && apt-get install -y python3.6 python3.6-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://get.docker.com/ | sh
# pip has to be installed before setuptools, setuptools has to be installed before tensorflow
RUN python3.6 -m pip install --no-cache-dir -U pip
RUN python3.6 -m pip install --no-cache-dir -U setuptools

RUN python3.6 -m pip install textworld==1.1.1 docker matplotlib torch
RUN python3.6 -m pip install -U spacy
RUN python3.6 -m spacy download en
RUN python3.6 -m pip install tensorboardx
RUN apt-get install -y libssl-dev
RUN python3.6 -m pip install allennlp

RUN ln -s /usr/bin/python3.6 /usr/bin/python
