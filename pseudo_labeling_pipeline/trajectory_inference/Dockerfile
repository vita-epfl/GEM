# Use an official Ubuntu image as the base
FROM docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04

# COPY /capstor/scratch/cscs/pmartell/trajectory_inference /app/trajectory_inference
WORKDIR /app/trajectory_inference
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
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-4

RUN MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" pip install -v -U git+https://github.com/facebookresearch/xformers.git@2bcbc55#egg=xformers
RUN pip3 install numpy opencv-python h5py scipy tensorboard p_tqdm
RUN pip3 install torch_scatter
RUN apt-get install libeigen3-dev -y

RUN cd /app/trajectory_inference/GeoCalib && python3 -m pip install -e .

RUN cd /app/trajectory_inference/droid_trajectory/droid_slam && \
    python3 setup.py install
    
RUN cd /app/trajectory_inference && \
    pip3 install -e .

RUN pip install omegaconf torchmetrics==0.10.3 fvcore iopath huggingface-hub==0.20.2 h5py pillow tqdm einops webdataset matplotlib deepspeed accelerate imageio imageio[ffmpeg] imageio[pyav] pytorch_lightning natsort omegaconf kornia clip@git+https://github.com/openai/CLIP.git open_clip_torch transformers scipy torchdata wandb moviepy imageio einx hydra-core more_itertools pandas vector-quantize-pytorch python-dotenv ultralytics lapx torchmetrics[image] scikit-learn

# Set up aliases for convenience
RUN echo 'alias python=python3' >> ~/.bashrc && \
echo 'alias pip=pip3' >> ~/.bashrc

# Entry point
CMD ["/bin/bash"]