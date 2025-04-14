# GEM's Pseudo-Labeling Pipeline

## Introduction

GEM’s Pseudo-Labeling Pipeline is a modular system for generating high-quality pseudo-labels for autonomous driving datasets. The pipeline consists of three key stages:
  
1. **Geometric Calibration**: Estimates camera intrinsics using [GeoCalib](https://github.com/cvg/GeoCalib).
2. **Depth Estimation**: Uses [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2) to predict scene depth.
3. **SLAM-based Trajectory Inference**: Leverages [DroidSLAM](https://github.com/princeton-vl/DROID-SLAM) to compute accurate camera trajectories.

## Environment setup

We provide a [Dockerfile](Dockerfile) to build an environment with all necessary dependencies. Ensure that [Docker](https://docs.docker.com/get-docker/) is installed on your system.

To build the Docker image, run:
```bash
docker build -t trajectory_inference .
```

## How to run

Download DepthAnythingV2's checkpoints from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) (outdoor version) and save them to a directory `./checkpoints`. Download the SLAM checkpoint from [here](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) and save it also to `./checkpoints`.

Before running inference, create a text file `file_list.txt` in the root directory, listing the paths to all `.h5` files (one per line).

Run first the calibration stage with e.g.
```bash
python3 scripts/geocalib_inference.py \
    --file-list "file_list.txt" \
    --log-dir "./logs" \
    --num-gpus 4 \
    --num-proc-per-gpu 1 \
    --batch-size 100 \
    --num-workers 24 \
    --camera-model "pinhole" \
    --no-profiler
```

Next, run the depth inference with e.g.
```bash
python3 scripts/depthanything_inference.py \
    --file-list "file_list.txt" \
    --weights-dir "./checkpoints" \
    --log-dir "./logs" \
    --num-gpus 4 \
    --num-proc-per-gpu 1 \
    --buffer-size 2048 \
    --batch-size 128 \
    --num-workers 24 \
    --encoder "vits" \
    --no-profiler
```

Finally, run the SLAM with e.g.
```bash
python3 scripts/droidslam_inference.py \
    --file-list "file_list.txt" \
    --weights "./checkpoints/droid.pth" \
    --log-dir "./logs" \
    --num-gpus 4 \
    --num-proc-per-gpu 1 \
    --trajectory-length 2000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 24 \
    --no-profiler
```

## Acknowledgements

- [Swiss Data Science Center](https://www.datascience.ch/) - for providing the pipeline's scripts.
- [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)
- [DroidSLAM](https://github.com/princeton-vl/DROID-SLAM)
- [GeoCalib](https://github.com/cvg/GeoCalib)
