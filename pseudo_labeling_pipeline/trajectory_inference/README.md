# Trajectory Inference

## Setup locally

Clone the repository:
```bash
git clone --recursive git@github.com:brdav/trajectory_inference.git
```

In an environment with Python>=3.9 and CUDA 12.1:
```bash
pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install opencv-python h5py scipy tensorboard
pip3 install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip3 install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
cd droid_trajectory/droid_slam
python3 setup.py install
```

Download the outdoor model checkpoints from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).

Download the droid.pth checkpoint from [here](https://github.com/princeton-vl/DROID-SLAM).


## Setup on Todi

[In progress...]

Make sure to set up credentials to be able to download containers from the Nvidia NGC catalog, see [here](https://confluence.cscs.ch/display/KB/LLM+Inference).

Clone the repository to `$SCRATCH`:
```bash
cd $SCRATCH
git clone --recursive git@github.com:brdav/trajectory_inference.git
```

Request an interactive job:
```
ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn srun -A a03 --time=1:00:00 --pty bash
```

Navigate to the project directory and build the Docker image:
```bash
cd $SCRATCH/trajectory_inference
podman build -v /capstor/scratch/cscs/pmartell/trajectory_inference:/app/trajectory_inference -t droid-slam .
enroot import -x mount -o $SCRATCH/droid-slam.sqsh podman://droid-slam
```

Copy the `.toml` file and replace `<user>` with username:
```bash
mkdir -p ~/.edf
cp trajectory-inference-env.toml ~/.edf/
vim ~/.edf/trajectory-inference-env.toml
```


## How to run

Before running inference, collect all h5 file paths (check script parameters):
```bash
srun -A a03 --environment=trajectory-inference-env --time=1:00:00 --pty bash
python scripts/collect_h5.py
exit
```

Check parameters in the `.sbatch` script, then submit with:
```bash
mkdir -p logs_slurm && for NODE_IDX in {0..0}; do ENROOT_LIBRARY_PATH="/capstor/scratch/cscs/fmohamed/enrootlibn" sbatch -A a03 --export=NODE_IDX=$NODE_IDX run_trajectory_inference.sbatch && sleep .5; done && sleep 5s && squeue -u pmartell
# mkdir -p logs_slurm
# for NODE_IDX in {0..63} ; do
#     nohup ./run_traj.sh > logs_slurm/out_${NODE_IDX}.log 2>&1 &
#     sleep .5
# done
sleep 5
squeue -u $USER
```


## How to undistort

It is recommended to work with undistorted images. A prerequisite to this is that the calib h5 file includes `distortion` parameters. If that's the case, we can create an undistorted version of that dataset, and then run the trajectory inference on that undistorted version. So far we should do it for ONCE and for DrivingDojo:
```bash
srun -A a03 --environment=trajectory-inference-env --time=12:00:00 --cpus-per-task=64 --pty bash
python scripts/undistort_h5.py \
    --base-dir /capstor/store/cscs/swissai/a03/datasets/ONCE \
    --replace-from ONCE \
    --replace-to ONCE_undistorted \
    --processes 64
python scripts/undistort_h5.py \
    --base-dir /capstor/store/cscs/swissai/a03/datasets/DrivingDojo_h5 \
    --replace-from DrivingDojo_h5 \
    --replace-to DrivingDojo_h5_undistorted \
    --processes 64
```
