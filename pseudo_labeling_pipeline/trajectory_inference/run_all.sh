mkdir -p logs_slurm
# 380
for NODE_IDX in {337..337} ; do
    echo "Running node index: $NODE_IDX"
    # ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn sbatch --export=NODE_IDX=${NODE_IDX} run_trajectory_inference.sbatch
    nohup ./run_traj.sh ${NODE_IDX} > logs_slurm/out_${NODE_IDX}.log 2>&1 &
    sleep .2
done