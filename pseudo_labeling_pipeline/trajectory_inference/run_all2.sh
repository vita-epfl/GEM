mkdir -p logs_slurm
# 0..552
for NODE_IDX in {0..552} ; do
    echo "Running node index: $NODE_IDX"
    # ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn sbatch --export=NODE_IDX=${NODE_IDX} run_trajectory_inference.sbatch
    nohup ./run_traj2.sh ${NODE_IDX} > logs_slurm/out_${NODE_IDX}.log 2>&1 &
    sleep .2
done
squeue -u $USER