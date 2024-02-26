FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated git ffmpeg libsm6 libxext6

RUN pip install opencv-python scipy shapely timm Polygon3 portalocker tabulate termcolor yacs matplotlib cloudpickle rapidfuzz==2.15.1 editdistance omegaconf protobuf absl-py google-auth google-auth-oauthlib grpcio markdown tensorboard-data-server werkzeug mxnet-mkl==1.6.0 numpy==1.23.1

COPY . /workspace/SwinTextSpotter

RUN cd /workspace/SwinTextSpotter && python setup.py build develop

