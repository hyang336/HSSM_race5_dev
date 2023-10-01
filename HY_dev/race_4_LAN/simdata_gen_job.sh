#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=ctb-akhanf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=HSSM_job
#SBATCH --output=/home/hyang336/jobs/HSSM%j.out

#submit simdata_gen.py 

source /home/hyang336/HSSM_race5_dev/HSSM_race5_dev_ENV/bin/activate
python /home/hyang336/HSSM_race5_dev/HY_dev/race_4_LAN/simdata_gen.py