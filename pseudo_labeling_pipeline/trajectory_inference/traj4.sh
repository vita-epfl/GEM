#!/bin/bash

echo "Installing geocalib..."
cd GeoCalib
python3 -m pip install -e .
cd ..
echo "Done."

pip3 install -e .

for GPU in {0..3} ; do
    echo "Running GPU index: $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/droidslam_inference.py \
        --file-list "$1" \
        --file-list-idx $GPU \
        --replace-from "/store/swissai/a03/datasets" \
        --replace-to "$SCRATCH/datasets" \
        --weights "/capstor/scratch/cscs/pmartell/trajectory_inference/weights/droid.pth" \
        --log-dir "./logs" \
        --num-gpus 1 \
        --num-proc-per-gpu 1 \
        --trajectory-length 2000 \
        --trajectory-overlap 100 \
        --min-trajectory-length 100 \
        --num-workers 24 \
        --exp-name "$2" \
        --no-profiler || true &
done
wait