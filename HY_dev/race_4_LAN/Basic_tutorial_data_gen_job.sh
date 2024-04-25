#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account=ctb-akhanf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --job-name=HSSM_datagen_job
#SBATCH --output=/home/hyang336/jobs/HSSM_datagen%j.out

module load gcc/9.3.0 cuda/11.8.0 cudnn/8.6 
export LD_LIBRARY_PATH=$EBROOTCUDA/lib:$EBROOTCUDNN/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

source /home/hyang336/HSSM_race5_dev/HSSM_race5_dev_ENV/bin/activate

echo $SLURM_JOBID

# Start a background process to save sstat results every 10 minutes
while true; do
  sstat -j $SLURM_JOBID --format=JobID,AveCPU,AveRSS,AveVMSize >> /home/hyang336/jobs/HSSM_datagen$SLURM_JOBID.out
  sleep 600
done &


python /home/hyang336/HSSM_race5_dev/HY_dev/race_4_LAN/Basic_tutorial.py
