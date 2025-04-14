#!/bin/bash
#SBATCH --job-name=trajectory_inference
#SBATCH --time=05:55:00
#SBATCH --nodes=1
#SBATCH --account=a03
#SBATCH --output=./logs_slurm/traj_%j.log
#SBATCH --error=./logs_slurm/error_traj_%j.log
#SBATCH --mem=460000
#SBATCH --environment=trajectory-inference-env \
#SBATCH --container-workdir=/capstor/scratch/cscs/pmartell/trajectory_inference

export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn

NUM_GPUS=4
FILE_PATHS_LIST="./output/file_list_node_${NODE_IDX}.txt"
EXP_NAME="traj_node_${NODE_IDX}"

srun python3 scripts/geocalib_inference.py \
    --file-list $FILE_PATHS_LIST \
    --replace-from "/store/swissai/a03/datasets" \
    --replace-to "${SCRATCH}/datasets \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 4 \
    --batch-size 50 \
    --num-workers 24 \
    --camera-model "pinhole" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python3 scripts/depthanything_inference.py \
    --file-list $FILE_PATHS_LIST \
    --replace-from "/store/swissai/a03/datasets" \
    --replace-to "${SCRATCH}/datasets \
    --weights-dir "/capstor/scratch/cscs/pmartell/trajectory_inference/weights" \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 4 \
    --buffer-size 2048 \
    --batch-size 64 \
    --num-workers 24 \
    --encoder "vits" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python3 scripts/droidslam_inference.py \
    --file-list $FILE_PATHS_LIST \
    --replace-from "/store/swissai/a03/datasets" \
    --replace-to "${SCRATCH}/datasets \
    --weights "/capstor/scratch/cscs/pmartell/trajectory_inference/weights/droid.pth" \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 1 \
    --trajectory-length 4000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 24 \
    --exp-name $EXP_NAME \
    --no-profiler

echo “”
echo “################################################################”
echo “@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@”
date
echo “@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@”
echo “################################################################”
