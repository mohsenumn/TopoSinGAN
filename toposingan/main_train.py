from config import get_arguments
from TopoSinGAN.manipulate import *
from TopoSinGAN.training import *
import TopoSinGAN.functions as functions
import torch
import random
import os, shutil
import time 

def generate_unique_dir(opt, seed_for_img_dir):
    while True:
        dir2save = functions.generate_dir2save(opt, seed_for_img_dir)
        if not os.path.exists(dir2save):
            return dir2save
        seed_for_img_dir = random.randint(1, 2000)
        opt = functions.post_config(opt, seed_for_img_dir)

if __name__ == '__main__':
    now = time.time()
    seed_for_img_dir = random.randint(1,2000)
    parser = get_arguments(seed_for_img_dir)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu_id', help='GPU ID to train')
    opt = parser.parse_args()
    opt = functions.post_config(opt, seed_for_img_dir)
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu_id))
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    dir2save = generate_unique_dir(opt, seed_for_img_dir)

    try:
        os.makedirs(dir2save)
    except OSError:
        pass
        
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp, seed_for_img_dir)
    then = time.time()
    print('****RUNTIME: ', then - now)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

