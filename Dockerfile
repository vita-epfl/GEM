FROM nvcr.io/nvidia/pytorch:24.05-py3
RUN nvcc --version
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
RUN python -c "import torch; print('Torch version:', torch.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"
RUN python -c "import triton"
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y software-properties-common && \
    apt-get install -y ninja-build && \
    apt-get install -y build-essential && \
    apt-get install -y cmake

RUN pip install --upgrade pip setuptools==69.5.1
RUN apt-get install -y openslide-tools
RUN apt-get install -y libpixman-1-0 build-essential cmake ninja-build ffmpeg

RUN python -c "import triton"

RUN python -c "import torch; print('Torch version:', torch.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"
RUN TORCH_CUDA_ARCH_LIST=90 MAX_JOBS=4 pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN python -c "import torch; print('Torch version:', torch.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"
RUN pip install \
        omegaconf \
        torchmetrics==0.10.3 \
        fvcore \
        iopath \
        huggingface-hub==0.20.2 \
        h5py \
        numpy \
        pillow \
        tqdm \
        einops \
        webdataset \
        matplotlib
RUN pip install deepspeed accelerate imageio imageio[ffmpeg] imageio[pyav] pytorch_lightning natsort omegaconf kornia clip@git+https://github.com/openai/CLIP.git open_clip_torch transformers scipy torchdata wandb moviepy imageio
RUN pip install einx hydra-core more_itertools pandas vector-quantize-pytorch python-dotenv

RUN python -c "import torch; print('Torch version:', torch.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"
RUN python -c "import xformers"
RUN python -c "import triton"