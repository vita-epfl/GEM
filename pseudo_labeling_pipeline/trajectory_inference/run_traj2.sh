#!/bin/bash

# Job parameters
JOB_NAME="trajectory_inference"
TIME_LIMIT="12:00:00"
ENVIRONMENT="trajectory-inference-env"
NODES=1
ACCOUNT="a03"
OUTPUT_LOG="./logs_slurm/traj_$(date +%Y%m%d_%H%M%S).log"
WORKDIR="/capstor/scratch/cscs/$USER/trajectory_inference"
MEMORY=460000
NUM_GPUS=4

# Set file paths and experiment names based on node index
NODE_IDX=$1  # Default to 0 if NODE_IDX is not set
FILE_PATHS_LIST="./output/file_list_node_${NODE_IDX}.txt"
EXP_NAME="traj_node_${NODE_IDX}"

echo "Running node index: $NODE_IDX"

# Change to the working directory

srun --nodes=$NODES \
    --environment="$ENVIRONMENT" \
    --account=a03 \
    --mem=460000 \
    --ntasks-per-node=1 \
    --time="$TIME_LIMIT" \
    ./prep.sh python3 scripts/geocalib_inference.py \
    --file-list "$FILE_PATHS_LIST" \
    --log-dir "./logs" \
    --num-gpus "$NUM_GPUS" \
    --num-proc-per-gpu 1 \
    --batch-size 100 \
    --num-workers 24 \
    --camera-model "pinhole" \
    --exp-name "$EXP_NAME" \
    --replace-from "/store/swissai/a03/datasets/" \
    --replace-to "/capstor/scratch/cscs/pmartell/datasets/" \
    --no-profiler || true

# # Run the second script
srun --nodes=$NODES \
    --environment="$ENVIRONMENT" \
    --account=a03 \
    --mem=460000 \
    --ntasks-per-node=1 \
    --time="$TIME_LIMIT" \
    ./prep.sh python3 scripts/depthanything_inference.py \
    --file-list "$FILE_PATHS_LIST" \
    --weights-dir "/capstor/scratch/cscs/pmartell/trajectory_inference/weights" \
    --log-dir "./logs" \
    --num-gpus "$NUM_GPUS" \
    --num-proc-per-gpu 1 \
    --buffer-size 2048 \
    --batch-size 128 \
    --num-workers 24 \
    --encoder "vits" \
    --replace-from "/store/swissai/a03/datasets/" \
    --replace-to "/capstor/scratch/cscs/pmartell/datasets/" \
    --exp-name "$EXP_NAME" \
    --no-profiler || true


# # # Run the third script
srun --nodes=$NODES \
     --environment="$ENVIRONMENT" \
     --account=a03 \
     --mem=460000 \
     --ntasks-per-node=1 \
     --time="$TIME_LIMIT" \
     ./traj4.sh $FILE_PATHS_LIST $EXP_NAME

# Completion message
# echo ""
# echo "################################################################"
# echo "@@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
# date
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "################################################################"