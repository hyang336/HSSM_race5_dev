#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=ctb-akhanf
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_valid_MLP_r5nb
#SBATCH --output=/home/hyang336/jobs/r5nb%j.out

#load GPU modules and set env variables
ml StdEnv/2020
module load gcc/9.3.0 cuda/11.8.0 cudnn/8.6 
export LD_LIBRARY_PATH=$EBROOTCUDA/lib:$EBROOTCUDNN/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

source /home/hyang336/HSSM_race5_dev/HSSM_race5_dev_ENV/bin/activate

python /home/hyang336/HSSM_race5_dev/HY_dev/race_5nb_LAN/train_valid_MLP.py
