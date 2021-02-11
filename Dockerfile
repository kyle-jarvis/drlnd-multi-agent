FROM ubuntu:18.04

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    wget ca-certificates vim git gdb \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf \
    libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
    

RUN useradd --create-home user
USER user
WORKDIR /home/user

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

COPY conf/extra_requirements.txt /home/user/bootstrap/

RUN conda create --name drlnd python=3.6 && \
    eval "$(conda shell.bash hook)" && \
    conda init bash && \
    conda activate drlnd && \
    pip install gym['box2d'] && \
    pip install -e "git+https://github.com/udacity/deep-reinforcement-learning/#egg=unityagents&subdirectory=python" && \
    pip install -r "/home/user/bootstrap/extra_requirements.txt"

RUN touch /home/user/.bashrc && \
  echo 'export LC_ALL=C.UTF-8' >> /home/user/.bashrc && \
  echo 'export LANG=C.UTF-8' >> /home/user/.bashrc

ARG DISP_IP
ENV DISPLAY $DISP_IP