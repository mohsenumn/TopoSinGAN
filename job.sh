#!/bin/bash
#SBATCH --job-name=toposingan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=log_agricultural_fields.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
module load pytorch/2.2.0

start_time=$(date +%s)

python -u toposingan/main_train.py --input_name agricultural_fields.png --nc_z 4 --nc_im 4 --gpu_id 0

end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Execution time: $elapsed_time seconds"

