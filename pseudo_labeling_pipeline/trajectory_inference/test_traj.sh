#!/bin/bash

# Job parameters
JOB_NAME="trajectory_inference"
TIME_LIMIT="6:00:00"
ENVIRONMENT="trajectory-inference-env"
NODES=1
ACCOUNT="a03"
OUTPUT_LOG="./logs_slurm/TEST_traj_$(date +%Y%m%d_%H%M%S).log"
WORKDIR="/capstor/scratch/cscs/pmartell/trajectory_inference"
MEMORY=460000
NUM_GPUS=4

# Set file paths and experiment names based on node index
NODE_IDX=${NODE_IDX:-0}  # Default to 0 if NODE_IDX is not set
FILE_PATHS_LIST="./test_files.txt"
EXP_NAME="TEST_traj_node_${NODE_IDX}"

# Change to the working directory

srun --nodes=$NODES \
    --environment="$ENVIRONMENT" \
    --account=a03 \
    --mem=460000 \
    ./prep.sh python3 scripts/droidslam_inference.py \
    --file-list "$FILE_PATHS_LIST" \
    --weights "/capstor/scratch/cscs/pmartell/trajectory_inference/weights/droid.pth" \
    --log-dir "./logs" \
    --num-gpus "$NUM_GPUS" \
    --num-proc-per-gpu 1 \
    --trajectory-length 6000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 24 \
    --exp-name "$EXP_NAME" \
    --no-profiler || true

# Completion message
# echo ""
# echo "################################################################"
# echo "@@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
# date
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "################################################################"