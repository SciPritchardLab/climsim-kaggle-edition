FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    wget \
    curl \
    bzip2 \
    file \
    make \
    cmake \
    sudo \
    libbz2-dev \
    zlib1g-dev \
    libffi-dev \
    libatlas-base-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libsqlite3-dev \
    libreadline6-dev \
    libssl-dev \
    libncursesw5-dev \
    libdb-dev \
    libexpat1-dev \
    liblzma-dev \
    libgdbm-dev \
    libmpdec-dev \
    gcc \
    g++ \
    git \
    xz-utils \
    liblzma-dev \
    vim \
    cmake --fix-missing \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

############################
# python packages
############################
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /work
