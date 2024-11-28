FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV TZ=Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# Install software-properties-common and add deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

# Install required packages including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3.10-dev \
    git \
    ssh \
    vim \
    && apt-get clean

# Set default Python version
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install pip and upgrade setuptools
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools

# Install PyTorch and xformers
RUN python3 -m pip install torch==2.1.0 torchvision==0.16.0 \
    torchaudio==2.1.0 xformers==0.0.22.post4 \
    --index-url https://download.pytorch.org/whl/cu118

# Start SSH service
RUN service ssh restart

# Expose port
EXPOSE 8000
