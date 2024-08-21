import SinGAN.functions as functions
from SinGAN.functions import *
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
import numpy as np
from skimage.io import imsave, imread, imshow
import torch.nn.functional as F

class TopologicalLoss(nn.Module):
    def __init__(self):
        super(TopologicalLoss, self).__init__()
        
        self.kernel3 = torch.tensor([[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.kernel5 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    def soft_threshold(self, x, threshold, alpha=10.0):
        return torch.sigmoid(alpha * (-x + threshold))
    
    def forward(self, mask):
        loss = 0.0
        for kernel in [self.kernel3, self.kernel5]:
            if mask.is_cuda:
                kernel = kernel.cuda()
            if kernel.shape[2] == 3:
                pad = 1
            elif kernel.shape[2] == 5:
                pad = 2
                
            neighbors_count = F.conv2d(mask, kernel, padding=pad)
            line_ends = self.soft_threshold(neighbors_count, 1.0) * mask
            num_line_ends = line_ends.sum()
            num_line_pixels = mask.sum()
            l = num_line_ends / (num_line_pixels + 1e-8)
            loss += l
        return loss * 100

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalLoss22(nn.Module):
    def __init__(self):
        super(TopologicalLoss22, self).__init__()
        
        self.kernel1 = nn.Parameter(torch.tensor([ [1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        self.kernel01 = nn.Parameter(torch.tensor([[1, 1, 1],
                                      [0, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel02 = nn.Parameter(torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel03 = nn.Parameter(torch.tensor([[1, 1, 1],
                                      [1, 0, 0],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel04 = nn.Parameter(torch.tensor([[1, 0, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel05 = nn.Parameter(torch.tensor([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel06 = nn.Parameter(torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel07 = nn.Parameter(torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel08 = nn.Parameter(torch.tensor([[1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        
        self.kernel20 = nn.Parameter(torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel21 = nn.Parameter(torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                     [ 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel22 = nn.Parameter(torch.tensor([[1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel23 = nn.Parameter(torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel24 = nn.Parameter(torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.kernel3 = nn.Parameter(torch.Tensor(  [[0, 0, 1, 1], 
                       [0, 0, 0, 1], 
                       [1, 0, 0, 1],
                       [1, 1, 1, 1]]).view(1,1,4,4))

        self.kernel4 = nn.Parameter(torch.Tensor(  [[1, 1, 0, 0], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4))

        self.kernel5 = nn.Parameter(torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [0, 0, 1, 1]]).view(1,1,4,4))

        self.kernel6 = nn.Parameter(torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 0],
                              [1, 1, 0, 0]]).view(1,1,4,4))

        self.kernel7 = nn.Parameter(torch.Tensor(  [[1, 0, 0, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4))

        self.kernel8 = nn.Parameter(torch.Tensor(  [[1, 1, 1, 1], 
                              [0, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4))

        self.kernel9 = nn.Parameter(torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 0, 0, 1]]).view(1,1,4,4))

        self.kernel10 = nn.Parameter(torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 0],
                              [1, 1, 1, 1]]).view(1,1,4,4))
        self.kernels = nn.ParameterList([self.kernel01.to('cuda'), self.kernel02.to('cuda'), self.kernel03.to('cuda'), self.kernel04.to('cuda'), self.kernel05.to('cuda'), self.kernel06.to('cuda'), self.kernel07.to('cuda'), self.kernel08.to('cuda'), self.kernel1.to('cuda'), self.kernel20, self.kernel21, self.kernel22, self.kernel23, self.kernel24, self.kernel3, self.kernel4,self.kernel5, self.kernel6, self.kernel7,self.kernel8, self.kernel9, self.kernel10]).to('cuda')


    def soft_threshold(self, x, threshold, alpha=10.0):
        return torch.sigmoid(alpha * (-x + threshold))

    def endpoints(self, mask, kernel, padded=True):
        kernel_size = kernel.shape[-1]
        pad = 2
        if kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        neighbors_count = F.conv2d(mask, kernel, padding=pad).to('cuda')
        if kernel_size == 4:
          line_ends = self.soft_threshold(neighbors_count[:,:,:-1, :-1], 1.0) * mask
        else:
          line_ends = self.soft_threshold(neighbors_count, 1.0) * mask
        if padded:
          line_ends = line_ends[:, :, 1:-1, 1:-1]
        return line_ends

    def forward(self, mask):
        epss = torch.zeros_like(mask.to('cuda'))
        i = 1
        for kernel in self.kernels:
            if i in [2, 3, 4, 5, 6, 7, 8, 9]:
                padding = (1, 1, 1, 1)
                padded_mask = F.pad(mask[0], padding, 'constant', 1)
                eps = self.endpoints(padded_mask.unsqueeze(0).to('cuda'), kernel=kernel, padded=True)
                epss += eps
            i += 1
        loss = (epss[0] / 8).sum()
        return loss


class TopologicalLoss2(nn.Module):
    def __init__(self):
        super(TopologicalLoss2, self).__init__()
        self.kernel1 = torch.tensor([ [1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.kernel01 = torch.tensor([[1, 1, 1],
                                      [0, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel02 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel03 = torch.tensor([[1, 1, 1],
                                      [1, 0, 0],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel04 = torch.tensor([[1, 0, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel05 = torch.tensor([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel06 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel07 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel08 = torch.tensor([[1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        
        self.kernel20 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel21 = torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                     [ 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel22 = torch.tensor([[1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel23 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel24 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel3 = torch.Tensor(  [[0, 0, 1, 1], 
                       [0, 0, 0, 1], 
                       [1, 0, 0, 1],
                       [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel4 = torch.Tensor(  [[1, 1, 0, 0], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel5 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [0, 0, 1, 1]]).view(1,1,4,4)

        self.kernel6 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 0],
                              [1, 1, 0, 0]]).view(1,1,4,4)

        self.kernel7 = torch.Tensor(  [[1, 0, 0, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel8 = torch.Tensor(  [[1, 1, 1, 1], 
                              [0, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel9 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 0, 0, 1]]).view(1,1,4,4)

        self.kernel10 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 0],
                              [1, 1, 1, 1]]).view(1,1,4,4)
        self.kernels = [self.kernel01, self.kernel02, self.kernel03, self.kernel04, self.kernel05, self.kernel06, self.kernel07, self.kernel08, self.kernel1, self.kernel20, self.kernel21, self.kernel22, self.kernel23, self.kernel24, self.kernel3, self.kernel4,self.kernel5, self.kernel6, self.kernel7,self.kernel8, self.kernel9, self.kernel10]

    def soft_threshold(self, x, threshold, alpha=10.0):
        return torch.sigmoid(alpha * (-x + threshold))

    def endpoints(self, mask, kernel, padded=True):
        kernel_size = kernel.shape[-1]
        pad = 2
        if kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        neighbors_count = F.conv2d(mask, kernel, padding=pad)
        if kernel_size == 4:
          line_ends = self.soft_threshold(neighbors_count[:,:,:-1, :-1], 1.0) * mask
        else:
          line_ends = self.soft_threshold(neighbors_count, 1.0) * mask
        if padded:
          line_ends = line_ends[:, :, 1:-1, 1:-1]
        return line_ends

    def forward(self, mask):
        epss = torch.zeros_like(mask)
        i=1
        for kernel in self.kernels:
          if i in [2,3,4,5,6,7,8,9]:
            if mask.is_cuda:
                kernel = kernel.cuda()
            padding = (1, 1, 1, 1)
            padded_mask = F.pad(mask[0], padding, 'constant', 1)
            eps = self.endpoints(padded_mask.unsqueeze(0), kernel = kernel, padded=True)
            epss += eps
          i+=1
        loss = (epss[0]/8).sum()
        return loss


class CustSigmoid(nn.Module):
    def __init__(self, alpha=55, beta=0.5):
        super(CustSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x + self.beta)))



def total_variation_loss(image):
    # Calculate the difference between adjacent pixels
    diff_i = F.pad(image[:, :, :-1, :], (0, 1, 0, 0)) - image
    diff_j = F.pad(image[:, :, :, :-1], (0, 0, 0, 1)) - image

    # Square the differences and sum them up
    variation = torch.sum(torch.abs(diff_i)) + torch.sum(torch.abs(diff_j))

    return variation

def custom_loss(generated_image):
    # Edge Detection using Sobel operators
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(generated_image, sobel_x, padding=1)
    grad_y = F.conv2d(generated_image, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    # Binarize the gradient magnitude
    binary_image = (grad_magnitude > 0.5).float()  # Threshold can be tuned
    # Integrate the Total Variation loss
    tv_loss = total_variation_loss(binary_image)
    return tv_loss

# When training, you can combine this loss with other losses (e.g., adversarial loss) to train the GAN.
def train(opt,Gs,Zs,reals,NoiseAmp, seed_for_img_dir):
    print("seed_for_img_dir = ", seed_for_img_dir)
    real_ = functions.read_image(opt)
    #print("real_ ====", real_.shape)
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    #print("real 1 ===", real.shape)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt, seed_for_img_dir)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real = reals[len(Gs)]
    #print("real shape===", real.shape)
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device, dtype=torch.bool)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []
    # if len(Gs) > 3:
    #   niter = 2500
    # else:
    #   niter = opt.niter
    niter = opt.niter
    for epoch in range(niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            #print("z_opt shape===", z_opt.shape)
            z_opt = m_noise(z_opt.expand(1,opt.nc_z,opt.nzx,opt.nzy)) # # changed second paramter from 3 to opt.nc_z - Vajira
            #print("z_opt shape (after)===", z_opt.shape)
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,opt.nc_z,opt.nzx,opt.nzy)) # changed second paramter from 3 to opt.nc_z - Vajira
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output = netD(real).to(opt.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev
            
            fake = netG(noise.detach(),prev)
            

            torch.set_grad_enabled(True)
            
    
            output = netD(fake.detach())
#             errD_fake = output.mean() + poiD
            errD_fake = output.mean()
            
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()
            
        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            
            output = netD(fake)

            errG = -output.mean()
#             errG = (poi+1)*errG
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                if epoch % 500 == 0 or epoch == (opt.niter-1):
#                     plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev.float()), vmin=0, vmax=1)
                    plt.imsave('%s/z_opt.png' % (opt.outf), functions.convert_image_np(z_opt.float()), vmin=0, vmax=1)
#                     print('opt.noise_amp', opt.noise_amp)
#                     plt.imsave('%s/noise_amp_p_z_opt.png' % (opt.outf), functions.convert_image_np(opt.noise_amp*z_opt.float()), vmin=0, vmax=1)
                r_e_c_loss = loss(netG(Z_opt.detach(),z_prev),real)
                rec_loss = alpha*r_e_c_loss
                rec_loss.backward(retain_graph=True)
                # if len(Gs)<8 and epoch<1001:
                rec_loss = rec_loss.detach()

                # if epoch>1000 and len(Gs)>3:
                if len(Gs)>3:
                ### TOPO LOSS
                  topo_loss = TopologicalLoss2()
                  sigmoid_layer = CustSigmoid(alpha=55, beta=0.5)
                  xx = fake[:, 3:4, :, :]
                  siged = sigmoid_layer(xx)
                  tloss = topo_loss(siged)*10
                  # xx = fake[:, 2:3, :, :]
                  # siged = sigmoid_layer(xx)
                  # tloss2 = topo_loss(siged)
                  # tloss = (tloss1+tloss2)/10

                  tloss.backward(retain_graph=True)
                  if epoch == 1999:
                    torch.save(xx, f'params/temp/fake_xx_{len(Gs)}.pth')
                    torch.save(siged, f'params/temp/fake_siged_{len(Gs)}.pth')
                    if len(Gs)<7:
                      tloss = tloss.detach()
                else:
                  tloss = 0
            else:
                Z_opt = z_opt
                rec_loss = 0
                tloss = 0
            #print("Error is here...!")
            #errG.backward(retain_graph=True)
        optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            print('tloss = ',tloss, 'rec_loss = ', rec_loss, 'errG= ', errG)
            
        if epoch % 500 == 0 or epoch == (opt.niter-1):
#             imsave('%s/fake_xx.png' %  (opt.outf), xx)
#             imsave('%s/fake_xx1.png' %  (opt.outf), xx1)
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, opt.nc_z, z.shape[2], z.shape[3]) # changed the second parameter from 3 to opt.nc_z
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device, dtype=torch.bool)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
#     print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
#     print(netD)

    return netD, netG






