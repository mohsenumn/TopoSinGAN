#!/bin/bash -l
#SBATCH --job-name=oct28cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=16
#SBATCH --mem=10g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=ahmad178@umn.edu 
#SBATCH --gres=gpu:v100:8
#SBATCH -p v100

rm -f screenlog.0

screen -L python3.6 main_train.py --input_name new_gt_skel.png  --nc_z 4 --nc_im 4 --gpu_id 0;
# screen -L python3.6 main_train.py --input_name new_gt_skel.png  --nc_z 4 --nc_im 4 --not_cuda; 


