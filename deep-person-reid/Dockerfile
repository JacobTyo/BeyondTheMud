FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN pip install numpy cython h5py Pillow six scipy opencv-python matplotlib tb-nightly future yacs gdown flake8 yapf isort==4.3.21 imageio chardet wandb
