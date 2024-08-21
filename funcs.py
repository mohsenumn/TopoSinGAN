import os
import pandas as pd
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.morphology import skeletonize
import cv2 
import networkx as nx
import pymorph
import mahotas as mh
from scipy import spatial
from bresenham import bresenham
import matplotlib.pyplot as plt

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
def detect_tails2(mask, ep_val=3, bp_vals=[2], out_px_value=5, min_line_length=10):
    mask_bi = cv2.threshold(mask, 0, 1, 8)
    mask = mask_bi[1]
    skel = mh.thin(mask)
    skel = pruning(skel,3)
    bp = branchedPoints(skel)
    ep = endPoints(skel)
    edge = skel-bp
    bps = np.where(bp>0, 1, 0)
    eps = np.where(ep>0, 2, 0)
    combined = bps+eps+skel #ep_val becomes 3 here as a result of combination
    data = combined.copy()
    img_lines_removed = combined.copy()
    ep_coords = np.argwhere(data == ep_val)
    lines=dict()
    for p in ep_coords:
        lines.update({f'{list(p)}':[]})
    for point in ep_coords:
        center = point
        lines[f'{list(center)}'].append(list(point))
        no_bp = True
        counter = 0
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
            ker = [t, r, b, l]
            try:
                for j in ker:
                    if (data[j[0], j[1]] ==1) & (list(j) not in lines[f'{list(center)}']):
                        lines[f'{list(center)}'].append(list(j))
                        point = j
                    elif data[j[0], j[1]] in bp_vals: #branch points
                        no_bp = False
                if counter > len(data):
                    no_bp = False
            except:
                no_bp = False
    filterred_lines = {}
    eliminated_lines = {}
    for key in lines.keys():
        for p in lines[key]:
            if len(lines[key])>=min_line_length:
                data[p[0],p[1]] = out_px_value
                img_lines_removed[p[0],p[1]] = 0
                filterred_lines.update({key:lines[key]})
            else:
                data[p[0],p[1]] = 0
                img_lines_removed[p[0],p[1]] = 0
                ind = list(np.array(key.replace(' ', '').replace('[', '').replace(']', '').split(',')).astype(int))
                combined[ind[0], ind[1]]=4
                eliminated_lines.update({key:lines[key]})
    img_with_eps = combined.copy()
    img_lines_hlighted = data.copy()
    img_lines_removed = np.where(img_lines_removed>0, 1, 0)
    return img_with_eps, img_lines_hlighted, img_lines_removed #, filterred_lines, eliminated_lines

def detect_tails_iter(mask1, min_line_length=0, max_iter=2):
    mask1 = normalize(mask1, upper_bound=255, uint8 = True)
    mask2 = mask1*0.5
    i = 0
    flag = False
    while not flag:
        i+=1
        mask1 = mask2
        mask_n = normalize(mask1, upper_bound=255, uint8 = True)
        img_with_eps, img_lines_hlighted, img_lines_removed = detect_tails2(mask_n, min_line_length=0) 
        d1 = img_with_eps.copy()
        d1_2 = np.where(d1>4, 0,d1)
        d1_3 = np.where(d1_2>0, 1,0)
        mask2 = d1_3
        flag = np.array_equal(mask1, mask2)
        if i == max_iter:
            flag = True
    return mask2


def normalize_minmax(arr, lower_bound=0, upper_bound=255, type = None):
    if arr.max() != arr.min():
        if type != None:
            ratio = (upper_bound-lower_bound)/(np.max(arr)-np.min(arr))
            shift = (np.max(arr)+np.min(arr))/(upper_bound-lower_bound) 
            return ((arr - shift)*ratio).astype(type)
        else:
            ratio = (upper_bound-lower_bound)/(np.max(arr)-np.min(arr))
            shift = (np.max(arr)+np.min(arr))/(upper_bound-lower_bound) 
            return (arr - shift)*ratio
    else:
        if type != None:
            return arr.astype(type)
        else: 
            return arr

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
        ratio = False
        y = 1
#     yp = 1-(np.e**(-3*ratio/10))
    return [deg_0s, leaves, deg_2s, ratio, len(list(dici.values())), y]

def closeness_loss2(mask, file, min_line_length=0):
    img, data, img_lines_removed = detect_tails2(mask, ep_val=3, bp_vals=[2], out_px_value=5, min_line_length=min_line_length)
    leaves = len(np.where(img==3)[0]) # deg 1
    branch_points = len(np.where(img==2)[0]) # deg 3
    regular_points = len(np.where(img==1)[0]) #deg 2
    if branch_points != 0:
        ratio = (leaves/branch_points)*10
        y = 1-(np.e**(-ratio))
    else:
        y = 1
    return [file, leaves, regular_points, branch_points, y]


def normalize(arr, upper_bound=255, uint8 = True):
    if uint8:
        return ((arr - arr.min())*(upper_bound/(arr.max()-arr.min()))).astype('uint8')
    else:
        return (arr - arr.min())*(upper_bound/(arr.max()-arr.min()))


def directory_poi(directory, mask_band, plot=False, rank=0):
    """
    Arguments: 
                directory: directory to the input image 
                mask_band: the number of the band which contains the mask
                
    Returns: a Panda Dataframe sorted by the POI value ascending. It contains ['name', 'deg0', 'deg1', 'deg2', 'POI']
    """
    pd.set_option('display.max_rows', None)
    cwd = os.getcwd()
    directory = cwd + directory.split(cwd.split('/')[-1])[-1]
    pois2 = []
    files_in_directory = os.listdir(directory)
    for file in files_in_directory:
        if file.endswith("mask.png"): 
            path_to_file = os.path.join(directory, file)
            im = imread(path_to_file)
            c = im[:,:,mask_band]
            poi = closeness_loss2(normalize(c), file[:-4])
            pois2.append(poi)
    
    pdf = pd.DataFrame(pois2, columns = ['name', 'deg0', 'deg1', 'deg2', 'POI'])
    pdf.reset_index(inplace=True)
    sorted_df = pdf.sort_values(by='POI').reset_index(drop=True)
    if plot:
        top_file = os.path.join(directory, sorted_df.loc[rank,'name']+'.png')
        print(f'{rank}th Mask With The Lowest POI: ', sorted_df.loc[rank,'name']+'.png', 'POI= ', sorted_df.loc[rank,'POI'])
        imshow(top_file)
        plt.show()
    return sorted_df

def plotter(directory, linspace_len=None, quantity=None, scale=None, top=0.7, bottom=0.4, hspace=0.7, suffix='mask'):
    """
    Gets a global directory and plots all figures along with their names.
    """
    cwd = os.getcwd()
    directory = cwd + directory.split(cwd.split('/')[-1])[-1]
    files_in_directory = os.listdir(directory)
    if linspace_len == None:
        for file in files_in_directory:
            if file.endswith(f"{suffix}.png"): 
                path_to_file_or_folder = os.path.join(directory, file)
                print(path_to_file_or_folder.split('/')[-1])
                imshow(path_to_file_or_folder)
                plt.show()
    else:
        init_val = scale * linspace_len
        title = f'scale {scale}'
        msks = []
        rgbs = []
        for file in files_in_directory:
            if file.endswith("mask.png"): 
                path_msk = os.path.join(directory, file)
                msks.append(path_msk)
            elif file.endswith("img.png"): 
                path_rgb = os.path.join(directory, file)
                rgbs.append(path_rgb)  
        msks = sorted(msks)
        rgbs = sorted(rgbs)
        tickers = np.linspace(init_val, linspace_len + init_val - 1, quantity).astype(int)
        fig, axs = plt.subplots(2, int(quantity/2), figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.suptitle(title, fontsize=18, y=0.79)
        plt.subplots_adjust(bottom=bottom, top=top, hspace=hspace)
        axs = axs.ravel()
        for i, ax in zip(range(quantity), axs.ravel()):
            axs[i].imshow(imread(msks[tickers[i]]))
            axs[i].set_title(msks[tickers[i]].split('/')[-1])
    plt.show()
            
def instance_plotter(directory, index, band, name = None, plot=True, rgb = False, figsize=7):
    """
    Gets a global directory and plots the bandth band of the indexth image along with its name.
    Returns the image
    """
    cwd = os.getcwd()
    directory = cwd + directory.split(cwd.split('/')[-1])[-1]
    files_in_directory = os.listdir(directory)
    rgbs = []
    masks = []
    for file in files_in_directory:
        if file.endswith("mask.png"): 
            masks.append(file)
        elif file.endswith("img.png"):
            rgbs.append(file)
    if name != None:
        if name.endswith('.png'):
            pass
        else:
            name = name + '.png'
        index = masks.index(name)
    path_to_file_or_folder = os.path.join(directory, masks[index])
    im = imread(path_to_file_or_folder)[:,:,band]
    poi = closeness_loss2(normalize(im), masks[index])
    print(f"POI: {poi}")
    if rgb:
        path_to_rgb = os.path.join(directory, rgbs[index])
    if plot:
        plt.figure(figsize=(figsize,figsize))
        imshow(im)
        plt.show()
        if rgb:
            plt.figure(figsize=(figsize,figsize))
            imshow(path_to_rgb)
            plt.show()
    return im

def single_poi(path, band = 3, rgb = [0,1,2], plot_rgb=True):
    figsize=7
    """
    Gets a global path of an image and plots the bandth band of the indexth image along with its name.
    """
    cwd = os.getcwd()
    path = cwd + path.split(cwd.split('/')[-1])[-1]
    name = path.split('/')[-1]
    im = imread(path)
    mask = im[:,:,band]
    rgb = np.dstack([im[:,:,rgb[0]],im[:,:,rgb[1]],im[:,:,rgb[2]]])
    poi = closeness_loss2(normalize(mask), name)
    print(f"POI: {poi}")
    
    plt.figure(figsize=(figsize,figsize))
    imshow(mask)
    plt.show()
    if plot_rgb:
        plt.figure(figsize=(figsize,figsize))
        imshow(rgb)
        plt.show()
    return None