# TopoSinGAN
This repository contains the codes and materials for the TopoSinGAN research project. 

Instructions:
Navigate to the root folder of the repository and run the following command:

Using CUDA:

`python -u toposingan/main_train.py --input_name agricultural_fields.png --nc_z 4 --nc_im 4 --gpu_id 0`

Using CPUs:

`python -u toposingan/main_train.py --input_name agricultural_fields.png --nc_z 4 --nc_im 4 --not_cuda`


