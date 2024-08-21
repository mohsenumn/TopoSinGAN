#!/bin/bash
#SBATCH --job-name=bench2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --output=n1c12clover500x500.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
module load pytorch/2.2.0

start_time=$(date +%s)

# Execute the Python script
python -u /home/jacks.local/mohsen.ahmadkhani/imageprocessing/singan/singan/main_train.py --input_name clover500x500.png --nc_z 4 --nc_im 4 --gpu_id 0

# End time
end_time=$(date +%s)

# Calculate and display the elapsed time
elapsed_time=$((end_time - start_time))
echo "Execution time: $elapsed_time seconds"


