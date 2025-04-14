# Use an official Ubuntu image as the base
FROM docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app/gem_trajectory_inference
ENV DEBIAN_FRONTEND=noninteractive

# setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade pip setuptools

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y software-properties-common && \
    apt-get install -y ninja-build && \
    apt-get install -y build-essential && \
    apt-get install -y cmake
RUN apt-get install -y openslide-tools
RUN apt-get install -y libpixman-1-0 build-essential cmake ninja-build ffmpeg
RUN apt-get install -y wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-4

RUN MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" pip install -v -U git+https://github.com/facebookresearch/xformers.git@2bcbc55#egg=xformers
RUN pip3 install numpy opencv-python h5py scipy tensorboard p_tqdm
RUN pip3 install torch_scatter
RUN apt-get install libeigen3-dev -y

RUN cd /app/gem_trajectory_inference && \
    python3 -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"

RUN cd /app/gem_trajectory_inference/droid_trajectory && \
    python3 setup.py install
    
# Entry point
CMD ["/bin/bash"]