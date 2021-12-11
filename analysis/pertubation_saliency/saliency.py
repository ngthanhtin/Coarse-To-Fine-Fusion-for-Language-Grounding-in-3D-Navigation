# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

prepro = lambda img: imresize(img.mean(2), (80,80)).astype(np.float32)
searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size)
    mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def run_through_model(model, history, ix, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is None:
        im = history['ins'][ix]
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(history['ins'][ix].moveaxis(0, 2), mask) # perturb input I -> I'
        im = im.moveaxis(2, 0)
    
    obs = im
    # f = plt.figure(figsize=[30, 50])
    # plt.imshow(obs.moveaxis(0, 2))
    # for a in f.axes: a.get_xaxis().set_visible(False) ; a.get_yaxis().set_visible(False)
    # plt.show()
    # plt.close()

    instruction_idx = history['instruction'][ix-1]
    hx = history['hx'][ix-1]
    cx = history['cx'][ix-1]
    tx = history['tx'][ix-1]
    if blur_memory is not None: cx.mul_(1-blur_memory) # perturb memory vector

    value, logit, (hx, cx) = model( (torch.Tensor(obs.float().unsqueeze(0)),
                    torch.Tensor(instruction_idx), (tx, hx, cx)))
    return value if mode == 'critic' else logit

def score_frame(model, history, ix, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)

    scores = np.zeros((int(168/d)+1,int(300/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,168,d):
        for j in range(0,300,d):
            mask = get_mask(center=[i,j], size=[168,300], r=r)
            mask = np.expand_dims(mask, axis=2)
            mask = np.resize(mask, (168,300,3))
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data.numpy()
    
    pmax = scores.max()
    scores = imresize(scores, size=[168,300], interp='bilinear').astype(np.float32)
    return pmax * scores / scores.max()
    return scores

def saliency_on_vizdoom_frame(saliency, vizdoom, fudge_factor, channel=2, sigma=3):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = imresize(saliency, size=[168,300], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    I = vizdoom
    I[:,:,channel] += S
    
    I = I.clip(0.,1.)
    return I