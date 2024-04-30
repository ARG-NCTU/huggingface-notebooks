FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

###################################### user #####################################

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash \
    USER=arg \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} && \
    echo "root:root" | chpasswd && \
    echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

# Add the deadsnakes PPA for newer Python versions
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN apt-get install -y python3.11-venv && \
    python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip setuptools wheel

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \ 
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

##################################### PIP ######################################

RUN python3.11 -m pip install --upgrade pip setuptools && \
    python3.11 -m pip install \
    opencv-python \
    numpy \
    empy \
    tk \
    ale-py==0.8 \
    orjson \
    gymnasium \
    gymnasium[atari] \
    gymnasium[accept-rom-license] \
    matplotlib \
    pandas \
    torch \
    torchaudio \
    torchvision \
    quart \
    rich \
    opencv-python \
    scikit-learn \
    swig \
    mypy \
    transformers \
    datasets \
    evaluate \
    transformers[sentencepiece] \
    accelerate \
    timm \
    albumentations \
    roma \
    jupyter \
    ipytest \
    && apt-get update \
    && apt-get install -y python3.11-dev

RUN python3.11 -m pip install \
    gym \
    stable-baselines3 \
    gymnasium[box2d] 

##################################### setting ###################################################

RUN chown -R ${USER}:${USER} ${HOME}/ && \
    echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]
