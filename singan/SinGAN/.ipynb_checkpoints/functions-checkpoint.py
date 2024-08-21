import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans

from skimage.draw import line
from skimage.io import imread, imshow
from skimage.morphology import skeletonize
import cv2 
import networkx as nx
#########################
####### Mohsen's ADDITION ########
#########################
# import pymorph
# import mahotas as mh

def topologize(mask1, max_iter):
    mask1 = normalize(mask1, upper_bound=255, uint8 = True)
    mask2 = mask1*0.5
    i = 0
    flag = False
    while not flag:
        i+=1
        mask1 = mask2
        mask_n = normalize(mask1, upper_bound=255, uint8 = True)
        d1 = detect_tails(mask_n, ep_val=3, bp_vals=[2], out_px_value=5) 
        d1_2 = np.where(d1>4, 0,d1)
        d1_3 = np.where(d1_2>0, 1,0)
        mask2 = d1_3
        flag = np.array_equal(mask1, mask2)
        if i == max_iter:
            flag = True
    return mask2

def topo_tensor(fake, opt, max_iter=20):
    xx = convert_image_np(fake.detach())
    xx1 = normalize(xx[:,:,-1])
#     topo_err = detect_tails(xx1, ep_val=3, bp_vals=[2], out_px_value=5) 
    topo_err = topologize(xx1, max_iter)
#     one = normalize(xx[:,:,0]+topo_err)
#     two = normalize(xx[:,:,1]+topo_err)
#     three = normalize(xx[:,:,2]+topo_err)
    one = xx[:,:,0]+topo_err
    two = xx[:,:,1]+topo_err
    three = xx[:,:,2]+topo_err
    all_bands = np.dstack((one, two, three, topo_err)) #diff is here
    all_bands = np.rollaxis(all_bands, 2)
    all_bands = np.expand_dims(all_bands, axis=0)
    t = torch.Tensor(all_bands)
    x = move_to_gpu(t, opt.device)
    return x
        
def branchedPoints(skel):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mh.morph.hitmiss(skel,branch1)
    br2=mh.morph.hitmiss(skel,branch2)
    br3=mh.morph.hitmiss(skel,branch3)
    br4=mh.morph.hitmiss(skel,branch4)
    br5=mh.morph.hitmiss(skel,branch5)
    br6=mh.morph.hitmiss(skel,branch6)
    br7=mh.morph.hitmiss(skel,branch7)
    br8=mh.morph.hitmiss(skel,branch8)
    br9=mh.morph.hitmiss(skel,branch9)
    return br1+br2+br3+br4+br5+br6+br7+br8+br9

def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    endpoint9=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep9=mh.morph.hitmiss(skel,endpoint9)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8+ep9
    return ep
def pruning(skeleton, size):

    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

def shuffle(a):
    a = list(a)
    random.shuffle(a)
    return a 
 
def detect_tails(mask, ep_val=3, bp_vals=[2], out_px_value=5):
#     path = 'Input/Images/new_gt.png'
#     mask = imread(path)[:,:,band]
#     d1 = detect_tails(mask, ep_val=3, bp_vals=[2], out_px_value=5) 
    mask_bi = cv2.threshold(mask, 0, 1, 8)
    mask = mask_bi[1]

    img = mask

    skel = mh.thin(img)
    skel = pruning(skel,3)
    bp = branchedPoints(skel)
    ep = endPoints(skel)
    edge = skel-bp

    bps = np.where(bp>0, 1, 0)
    eps = np.where(ep>0, 2, 0)
    combined = bps+eps+skel

    data = combined.copy()
    # data = np.pad(data, 3, mode='constant')

    # bp_coords = np.argwhere(data ==2)
    ep_coords = np.argwhere(data == ep_val)
    lines=dict()
    for p in ep_coords:
        lines.update({f'{list(p)}':[]})
    for point in ep_coords:
        center = point
        lines[f'{list(center)}'].append(list(point))
        no_bp = True
        counter = 0
        stuck = False
        while no_bp:

            counter+=1
            tl = [point[0]-1, point[1]-1]
            t = [point[0]-1, point[1]]
            tr = [point[0]-1, point[1]+1]
            l = [point[0], point[1]-1]
            c = [point[0], point[1]]
            r = [point[0], point[1]+1]
            bl = [point[0]+1, point[1]-1]
            b = [point[0]+1, point[1]]
            br = [point[0]+1, point[1]+1]
            kernel = [[tl, t, tr],
                      [l, c, r],
                      [bl, b, br]]
            try:
                for i in range(len(kernel)):
                    for j in range(len(kernel)):
                        if (data[kernel[i][j][0], kernel[i][j][1]] ==1) & (list(kernel[i][j]) not in lines[f'{list(center)}']):
                            lines[f'{list(center)}'].append(list(kernel[i][j]))
                            point = kernel[i][j]
                        elif data[kernel[i][j][0], kernel[i][j][1]] in bp_vals: #branch points
                            no_bp = False
                if counter > len(data):
                    no_bp = False
            except:
                no_bp = False
                
    for key in lines.keys():
        for p in lines[key]:
            data[p[0],p[1]] = out_px_value
    return data



# custom weights initialization called on netG and netD

def read_image(opt): # there is an another function with same name, check below : vajira
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    #print("read image x : =====", x.shape)
    return np2torch(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())
def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def convert_image_np(inp):
    if inp.shape[1]==3 or inp.shape[1] == 4:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    inp=np.ascontiguousarray(inp)
#     imsave(f'fake_during/{random.randint(0, 2000)}.png', inp)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,device,num_samp=1,type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t, device):
    if (torch.cuda.is_available()):
        t = t.to(device)
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

#     print('interpolates', interpolates)
    
    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


################################################################
####### new loss function (polygon closeness) poi POI###################
################################################################



def skel_to_graph(skeIm, connectivity=2):
    assert(len(skeIm.shape) == 2)
    skeImPos = np.stack(np.where(skeIm))
    skeImPosIm = np.zeros_like(skeIm, dtype=np.int32)
    skeImPosIm[skeImPos[0], skeImPos[1]] = np.arange(0, skeImPos.shape[1])
    g = nx.Graph()
    if connectivity == 1:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    elif connectivity == 2:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
#     elif connectivity == 3:
#         neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1], 
#                          [2,2],[2,1],[2,0],[2,-1],[2,-2],[-2,2],[-2,1],[-2,0],[-2,-1],[-2,-2],
#                           [-1,-2],[0,-2],[1,-2],[-1,2],[0,2],[1,2]])
    else:
        raise ValueError(f'unsupported connectivity {connectivity}')
    for idx in range(skeImPos[0].shape[0]):
        for neighIdx in range(neigh.shape[0]):
            curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
            if np.any(curNeighPos<0) or np.any(curNeighPos>=skeIm.shape):
                continue
            if skeIm[curNeighPos[0], curNeighPos[1]] > 0:
                g.add_edge(skeImPosIm[skeImPos[0, idx], skeImPos[1, idx]], skeImPosIm[curNeighPos[0], curNeighPos[1]], weight=np.linalg.norm(neigh[neighIdx]))
#     g.graph['physicalPos'] = skeImPos.T
    return g


def skeletonizer(img):
#     (thresh, bw) = cv2.threshold(img[:,:,2], 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh, bw) = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    skeleton_lee = skeletonize(bw, method='lee')
    return skeleton_lee

def closeness_loss(img):
    skeleton = skeletonizer(img)
    G = skel_to_graph(skeleton, connectivity=2)
    deg_view = G.degree
    dici = dict(deg_view)
    deg_0s = list(dici.values()).count(0)
    leaves = list(dici.values()).count(1) #count the nodes having a degree of 1 (leaf nodes)
    deg_2s = list(dici.values()).count(2)
    if deg_2s != 0:
        ratio = (leaves/deg_2s)*10
        y = 1-(np.e**(-6*ratio))
    else:
#         ratio = 0
        y = 1
#     y = 1-(np.e**(-6*ratio))
#     yp = 1-(np.e**(-3*ratio/10))
    return y
# im_path = 'gt.png'
# closeness_loss(imread(im_path))
################################################################
# Linear Noise 
################################################################

def linear_noise(dims, n_lines, rng=255):
    '''
    arguments: 
        dims (tuple or list): the shape of the desired output random line image. 
        n_lines (integer): number of random lines in the image
        rng (integer; usually 255 or 1): the range of the pixel values. If 1 is specified, 
                                            the output image will have random values between 0 and 1. 
    '''
    
    def rand(rng):
        if rng ==1:
            return random.random()
        else:
            return random.randint(0,rng)
        
        
    if rng > 1:
        rand_img = np.zeros((dims[0], dims[1]), dtype=np.uint8)
    else:
        rand_img = np.zeros((dims[0], dims[1]), dtype=float)

    for i in range(n_lines):
        try:
            rr, cc = line(rand(dims[0]), rand(dims[1]), rand(dims[0]), rand(dims[1]))
            rand_img[rr, cc] = rand(rng)

        except:
            continue
    return rand_img

#generate a numpy array of shape input_img.shape with 500 line with pixel values between 0 and 255
# linear_noise(input_img.shape, 500, 255) 

################################################################




def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))

#     print("X shape=", x.shape)
    if len(x.shape) == 2: # executee if imag is gray
        x = color.gray2rgb(x)
#     print("x after imread===", x.shape)
    x = np2torch(x,opt)
    #print("x after np2torch===", x.shape)
    #print(" x.shape[1]=", x.shape[1])
    if x.shape[1] == 4:
        x = x[:,0:4,:,:]    
    else:
        x = x[:,0:3,:,:]
    #print("x in read image===", x.shape)
    return x

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2torch(x,opt):
    if opt.nc_im == 3 or opt.nc_im == 4: # added opt.nc_im == 4 by vajira to handle 4 channel image
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x, opt.device)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    if real.shape[1]==4:
        real = real[:,0:4,:,:] # added by vajira
    else:
        real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model exists, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device, dtype=torch.bool)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt, seed_for_img_dir):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = f'params/TrainedModels/TrainedModels_{seed_for_img_dir}/{opt.input_name[:-4]}/scale_factor={opt.scale_factor_init},alpha={opt.alpha}' 
    elif (opt.mode == 'animation_train') :
        dir2save = f'params/TrainedModels/TrainedModels_{seed_for_img_dir}/{opt.input_name[:-4]}/scale_factor={opt.scale_factor_init}_noise_padding' 
    elif (opt.mode == 'paint_train') :
        dir2save = f'params/TrainedModels/TrainedModels_{seed_for_img_dir}/{opt.input_name[:-4]}/scale_factor={opt.scale_factor_init}_paint/start_scale={opt.paint_start_scale}'
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt, seed_for_img_dir):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu_id))
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = f'TrainedModels_{seed_for_img_dir}/{opt.input_name[:-4]}/scale_factor={opt.scale_factor}/'
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    #print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def normalize255(array):
    return np.rint((array - np.nanmin(array)) * (255 / (np.nanmax(array) - np.nanmin(array))))

def normalize(arr, upper_bound=255, uint8 = True):
    if uint8:
        return ((arr - arr.min())*(upper_bound/(arr.max()-arr.min()))).astype('uint8')
    else:
        return (arr - arr.min())*(upper_bound/(arr.max()-arr.min()))

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x, device)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers, device):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x,device)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask




