ARG from=ubuntu:focal
FROM ${from}

# ========
# HEADLESS
# ========

# Change default shell to bash. This is effective only in the Dockerfile.
SHELL ["/bin/bash", "-i", "-c"]

# Create a new runtimeusers group and add root
RUN groupadd -K GID_MIN=100 -K GID_MAX=499 runtimeusers &&\
    gpasswd -a root runtimeusers

# Execute commands as root:runtimeusers so that any user created during runtime has rights
# to operate on the filesystem, and particularly the conda environment
USER root:runtimeusers

# Setup locales and timezone
ARG TZ=Europe/Rome
ARG DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/localtime &&\
    ln -s /usr/share/zoneinfo/"${TZ}" /etc/localtime &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends locales locales-all tzdata &&\
    rm -rf /var/lib/apt/lists/*


# System utilities
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        software-properties-common \
        apt-transport-https \
        apt-utils \
        git \
        wget \
        nano \
        bash-completion \
        gnupg2 \
        colordiff \
        curl \
        zip \
        unzip \
        lsof \
        net-tools \
        iputils-ping \
        strace \
        less \
        tree \
        htop \
        libopenblas-base \
        tmux \
        &&\
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ENV CONDA_PREFIX=/conda
ARG CONDA_PYTHON_VERSION=3.8
ENV MAMBA_ROOT_PREFIX=$CONDA_PREFIX/.mamba

# Install micromamba and create conda environment
RUN cd /usr/local &&\
    wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest \
        | tar -xvj bin/micromamba &&\
    eval "$(micromamba shell hook -s bash)" &&\    
    micromamba create -y -p $CONDA_PREFIX "python==$CONDA_PYTHON_VERSION.*" mamba -c conda-forge &&\
    micromamba activate $CONDA_PREFIX &&\
    conda config --system --add channels conda-forge &&\
    find $CONDA_PREFIX -group runtimeusers -not -type l -perm /u+w -not -perm -g+w -print -exec chmod g+w '{}' + &&\
    conda clean -afy

# Enable by default the conda environment for all users
RUN echo 'function activate_conda() {' >> /etc/bash.bashrc &&\
    echo '  eval "$(micromamba shell hook -s bash)"' >> /etc/bash.bashrc &&\
    echo '  micromamba activate $CONDA_PREFIX' >> /etc/bash.bashrc &&\
    echo '}' >> /etc/bash.bashrc &&\
    echo '[[ -z $NO_CONDA ]] && activate_conda' >> /etc/bash.bashrc

# Install buildchain
RUN find $CONDA_PREFIX -group runtimeusers -not -type l -perm /u+w -not -perm -g+w -print -exec chmod g+w '{}' + &&\
    conda clean -afy

# Packages installed later switch blas implementation to mkl, pinning it from the beginning here
#RUN echo 'libblas=*=*mkl' >> $CONDA_PREFIX/conda-meta/pinned 
# ATTENTION !!!! Use the aforcommand if you do not want a lirbary to be changes (to be used for numpy )

# Default directory with sources
ARG SRC_DIR=/usr/local/src

USER root:root

# Change default shell to bash. This is effective only in the Dockerfile.
SHELL ["/bin/bash", "-i", "-c"]
RUN conda config --append channels robotology

RUN mamba install _openmp_mutex=4.5
COPY conda_requirements.txt .
RUN mamba install -y --file  conda_requirements.txt

    
COPY pip_requirements.txt .
RUN pip install -r pip_requirements.txt

#Get coinbrew
# Install HSL solvers
ADD coinhsl.zip /tmp/coinhsl.zip
RUN mamba install compilers make pkg-config libblas liblapack metis && \
    cd /tmp &&\
    git clone --depth=1 -b releases/2.2.1 https://github.com/coin-or-tools/ThirdParty-HSL &&\
    cd ThirdParty-HSL &&\
    unzip /tmp/coinhsl.zip &&\
    ln -s coinhsl-* coinhsl &&\
    ./configure --prefix=$CONDA_PREFIX &&\
    make install &&\
    cd $CONDA_PREFIX/lib  &&\
    ln -s -v ./libcoinhsl.so ./libhsl.so &&\
    rm -r /tmp/coinhsl* &&\
    rm -r /tmp/ThirdParty-HSL


RUN mkdir genetic_algorithm_code
WORKDIR /genetic_algorithm_code/
COPY . . 
