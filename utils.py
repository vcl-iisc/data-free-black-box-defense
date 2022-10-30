import os
import sys
import time
import math
import pywt
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.alexnet import Alexnet
from models.half_alexnet import HalfAlexnet

from skimage.restoration import denoise_wavelet

from models.resnet import BlackBoxResNet18, ResNet18, ResNet34
from models.wrn import build_WideResNet
from models.vgg import VGG
from pytorch_wavelets import DWTForward, DWTInverse

from wavelet_denoising import bayes_threshold, get_var, rig_sure # (or import DWT, IDWT)

#from util import frequencyHelper 
#from frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t
'''_______________________________________________________'''
import numpy as np
from scipy import signal

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([32, 32]), r) ## Since I rescaled MNSIT images to 32x32 to use CIFAR modified models directly
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([32, 32]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([32 * 32]))


    Images_freq_high = []
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([32, 32]))
        fd = fd * (1-mask)
        img_high = ifftshift(fd)
        Images_freq_high.append(np.real(img_high).reshape([32 * 32]))

    return np.array(Images_freq_low), np.array(Images_freq_high)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)


def torch_fftshift(img):
    fft = torch.fft.fft2(img)
    return torch.fft.fftshift(fft)

def torch_ifftshift(fft):
    fft = torch.fft.ifftshift(fft)
    return torch.fft.ifft2(fft)


masks=None
def generateDataWithDifferentFrequencies_3Channel_new(Images,mask):

    mask = mask.unsqueeze(0).unsqueeze(0)
    fd = torch_fftshift(Images)
    
    low_freq_img = fd*mask
    low_freq_img = torch_ifftshift(low_freq_img)
    
    high_freq_img = fd*(1-mask)
    high_freq_img = torch_ifftshift(high_freq_img)

    return low_freq_img, high_freq_img


'''_______________________________________________________'''
def get_freq(data, mask):

    #images = data.detach().cpu()

    #images = images.permute(0,2,3,1)
    img_l, img_h = generateDataWithDifferentFrequencies_3Channel_new(data, mask = mask)
    #img_l, img_h = torch.from_numpy(np.transpose(img_l, (0,3,1,2))), torch.from_numpy(np.transpose(img_h, (0,3,1,2)))

    return img_l, img_h

class get_LL_mnist_torch(nn.Module) :
    def __init__(self,lev) :
        super(get_LL_mnist_torch, self).__init__()
        self.xfm = DWTForward(J=lev, mode='zero', wave='db1').cuda()
        self.ifm = DWTInverse(mode='zero', wave='db1').cuda()

    def forward(self, image):
        tmp1 = []
        n = image.size()[0]
        for i in range(n):
            imge = image[i,:,:,:]
            imge = imge.unsqueeze(0)
            tmp = []
            for i in range(1): 
                img = imge[:,i,:,:].unsqueeze(0)
                Yl, Yh = self.xfm(img)
                for i in range(len(Yh)):
                    zeros = torch.zeros_like(Yh[i]).cuda()
                    Yh[i] = Yh[i]*zeros
                tmp.append(self.ifm((Yl, Yh)))
            tmp1.append(torch.cat(tmp,axis=1))
        return torch.cat(tmp1,axis=0)
    
class get_LL_torch(nn.Module) :
    def __init__(self,lev) :
        super(get_LL_torch, self).__init__()
        self.xfm = DWTForward(J=lev, mode='zero', wave='db1').cuda()
        self.ifm = DWTInverse(mode='zero', wave='db1').cuda()

    def forward(self, image):
        tmp1 = []
        n = image.size()[0]
        for i in range(n):
            imge = image[i,:,:,:]
            imge = imge.unsqueeze(0)
            tmp = []
            for i in range(3): 
                img = imge[:,i,:,:].unsqueeze(0)
                Yl, Yh = self.xfm(img)
                for i in range(len(Yh)):
                    zeros = torch.zeros_like(Yh[i]).cuda()
                    Yh[i] = Yh[i]*zeros
                tmp.append(self.ifm((Yl, Yh)))
            tmp1.append(torch.cat(tmp,axis=1))
        return torch.cat(tmp1,axis=0)

def get_top_compfrom_HF2(image,keep=0.15,lev=1,wvlt='db1'):
    assert lev==2
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1))+list(coeffs[2][0].reshape(-1))+list(coeffs[2][1].reshape(-1))+list(coeffs[2][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new1 = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new1.append(coeffs[1][j]*ind)
        coeffs_new2 = []
        for j in range(3):
            ind = np.abs(coeffs[2][j]) > thresh
            coeffs_new2.append(coeffs[2][j]*ind)
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new1), tuple(coeffs_new2)], wavelet=wvlt))
    return tmp
 
def get_top_compfrom_HF3(image,keep=0.15,lev=1,wvlt='db1'):
    assert lev==3
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1))+list(coeffs[2][0].reshape(-1))+list(coeffs[2][1].reshape(-1))+list(coeffs[2][2].reshape(-1))+list(coeffs[3][0].reshape(-1))+list(coeffs[3][1].reshape(-1))+list(coeffs[3][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new1 = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new1.append(coeffs[1][j]*ind)
        coeffs_new2 = []
        for j in range(3):
            ind = np.abs(coeffs[2][j]) > thresh
            coeffs_new2.append(coeffs[2][j]*ind)
        coeffs_new3 = []
        for j in range(3):
            ind = np.abs(coeffs[3][j]) > thresh
            coeffs_new3.append(coeffs[3][j]*ind)
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new1), tuple(coeffs_new2), tuple(coeffs_new3)], wavelet=wvlt))
    return tmp
    
    
# def get_bottom_compfrom_HF(image,keep=0.15,lev=1,wvlt='db1'):
#     assert lev==2
#     image = image.detach().cpu()
#     tmp = []
#     for i in range(3): 
#         img = image[i,:,:]
#         coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
#         coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1))+list(coeffs[2][0].reshape(-1))+list(coeffs[2][1].reshape(-1))+list(coeffs[2][2].reshape(-1)))       
#         Csort = np.sort(np.abs(coeffs_HF))
#         thresh = Csort[int(np.floor(keep *len (Csort)))]
#         coeffs_new1 = []
#         for i in range(3):
#             ind = np.abs(coeffs[1][i]) < thresh
#             coeffs_new1.append(coeffs[1][i]*ind)
#         coeffs_new2 = []
#         for i in range(3):
#             ind = np.abs(coeffs[2][i]) < thresh
#             coeffs_new2.append(coeffs[2][i]*ind)
#         tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new1), tuple(coeffs_new2)], wavelet=wvlt))
#     return tmp

# def get_random_compfrom_HF(image,keep=0.15,lev=1,wvlt='db1'):
#     assert lev==2
#     image = image.detach().cpu()
#     tmp = []
#     for i in range(3): 
#         img = image[i,:,:]
#         coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
#         rand_a = np.zeros(48*16)
#         rand_a[:int(np.floor(keep *len (rand_a)))] = 1
#         np.random.shuffle(rand_a)
#         rand_a = rand_a.reshape((48,16))
#         coeffs[2][0][rand_a[:16,:]==0] = 0
#         coeffs[2][1][rand_a[16:32,:]==0] = 0
#         coeffs[2][2][rand_a[32:,:]==0] = 0
#         rand_b = np.zeros(24*8)
#         rand_b[:int(np.floor(keep *len (rand_b)))] = 1
#         np.random.shuffle(rand_b)
#         rand_b = rand_b.reshape((24,8))
#         coeffs[1][0][rand_b[:8,:]==0] = 0
#         coeffs[1][1][rand_b[8:16,:]==0] = 0
#         coeffs[1][2][rand_b[16:,:]==0] = 0
#         tmp                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                .append(pywt.waverec2(coeffs, wavelet=wvlt))
#     return tmp


def get_top_compfrom_HF(image,keep=0.15,lev=1,wvlt='db1'):
    assert lev==1
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
        coeffs_new = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new.append(coeffs[1][j]*ind)
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


def get_bottom_compfrom_HF(image,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor(keep *len (Csort)))]
        coeffs_new = []
        for i in range(3):
            ind = np.abs(coeffs[1][i]) < thresh
            coeffs_new.append(coeffs[1][i]*ind)
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp


def get_random_compfrom_HF(image,keep=0.15,lev=1,wvlt='db1'):
    assert lev==1
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        rand_a = np.zeros(48*16)
        rand_a[:int(np.floor(keep *len (rand_a)))] = 1
        np.random.shuffle(rand_a)
        rand_a = rand_a.reshape((48,16))
        coeffs[1][0][rand_a[:16,:]==0] = 0
        coeffs[1][1][rand_a[16:32,:]==0] = 0
        coeffs[1][2][rand_a[32:,:]==0] = 0
        tmp.append(pywt.waverec2(coeffs, wavelet=wvlt))
    return tmp

def replace_HF_with_mean(image,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_new = []
        for i in range(3):
            coeffs_new.append(np.ones((16,16))*np.mean(coeffs[1][i]))
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp

def replace_HF_with_max(image,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_new = []
        for i in range(3):
            coeffs_new.append(np.ones((16,16))*np.max(coeffs[1][i]))
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp

def get_comp_wv_expt10(image,keep=0.15,wvlt='db1'):
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=2)
        coeffs1 = pywt.wavedec2(img,wavelet=wvlt, level=1)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr*ind
        coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        coeffs_new = [coeffs1[0], coeffs_filt[2]]
        Arecon = pywt.waverec2(coeffs_new, wavelet=wvlt)
        tmp[i,:,:] = Arecon
    return tmp

def get_k_compfrom_HF_batch(image,setting,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = np.zeros(image.size())
    
    n = image.size()[0]
    for i in range(n):
        if setting=='top':
            if lev==1 :
                tmp1[i,:,:,:] = get_top_compfrom_HF(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
            if lev==2 :
                tmp1[i,:,:,:] = get_top_compfrom_HF2(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
            if lev==3:
                tmp1[i,:,:,:] = get_top_compfrom_HF3(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
        elif setting=='bottom':
            tmp1[i,:,:,:] = get_bottom_compfrom_HF(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
        elif setting=='random':
            tmp1[i,:,:,:] = get_random_compfrom_HF(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)   
        elif setting=='mean':
            tmp1[i,:,:,:] = replace_HF_with_mean(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)   
        elif setting=='max':
            tmp1[i,:,:,:] = replace_HF_with_max(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
        elif setting=='expt10':
            tmp1[i,:,:,:] = get_comp_wv_expt10(image[i,:,:,:],keep=keep,wvlt=wvlt)
        else:
            raise("wavelet coefficient selction setting not defined")
    return torch.from_numpy(tmp1) 

def get_comp_wv(image,keep=0.1,lev=3,wvlt='db1'):
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr*ind
        coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        Arecon = pywt.waverec2(coeffs_filt, wavelet=wvlt)
        tmp[i,:,:] = Arecon
    return tmp

def get_gn_perturbed_img(image,lev=1,wvlt='db1',mean=0., stddev=0.05):
#     assert lev==1
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(1): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        coeff_arr =  coeff_arr + np.random.randn(coeff_arr.shape[0], coeff_arr.shape[1])*stddev + mean
        coeffs_filt = pywt.array_to_coeffs (coeff_arr,coeff_slices, output_format='wavedec2')
        Arecon = pywt.waverec2(coeffs_filt, wavelet=wvlt)
        tmp[i,:,:] = Arecon
    return tmp

def get_gn_perturbed_img_batch(image,stddev=0.05,level=2):
    tmp1 = np.zeros(image.size())
    n = image.size()[0]
    for i in range(n):
        tmp1[i,:,:,:] = get_gn_perturbed_img(image[i,:,:,:],stddev=stddev,lev=level)
    tmp1 = torch.from_numpy(tmp1)
    return tmp1

def get_comp_wv_batch(image,keep=0.1,lev=3,wvlt='db1'):
    tmp1 = np.zeros(image.size())
    n = image.size()[0]
    for i in range(n):
        tmp1[i,:,:,:] = get_comp_wv(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
    return torch.from_numpy(tmp1)    


def get_ll_orig(image,keep=0.1,lev=1,wvlt='db1'): ## LL of level 1 
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        tmp.append(coeffs[0])
    return np.stack(tmp, axis=0)

def get_ll_orig_batch(image,lev=1):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_ll_orig(image[i,:,:,:],lev=lev))
    return torch.from_numpy(np.stack(tmp1,axis=0)) 

def get_ll_orig_inverse(image,keep=0.1,lev=1,wvlt='db1'):
    image = image.detach().cpu()
    tmp = tuple([np.zeros(image[0,:,:].shape)]*3)
    img = []
    for i in range(3): 
        coeffs = [image[i,:,:], tmp] 
        img.append(pywt.waverec2(coeffs, wavelet=wvlt))
    return np.stack(img, axis=0)

def get_ll_orig_inverse_torch(args, image):
    tmp = [torch.zeros(1,1,3,16,16).to(args.device)]
    img = []
    ifm = DWTInverse(mode='zero', wave='db1').to(args.device)
    for i in range(3): 
        coeffs = ((image[i,:,:].unsqueeze(0)).unsqueeze(0), tmp) 
        img.append(ifm(coeffs))
    return torch.cat(img, axis=1)
def get_ll_orig_inverse_torch_batch(args, image):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_ll_orig_inverse_torch(args, image[i,:,:,:]))
    return torch.cat(tmp1,axis=0)

def get_ll_orig_inverse8(image,keep=0.1,lev=1,wvlt='db1'):
    image = image.detach().cpu()
    tmp = tuple([np.zeros(image[0,:,:].shape)]*3)
    tmp2 = tuple([np.zeros((16,16))]*3)
    img = []
    for i in range(3): 
        coeffs = [image[i,:,:], tmp, tmp2] 
        img.append(pywt.waverec2(coeffs, wavelet=wvlt))
    return np.stack(img, axis=0)
def get_ll_orig_inverse_batch8(image):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_ll_orig_inverse8(image[i,:,:,:]))
    return torch.from_numpy(np.stack(tmp1,axis=0)) 

def get_ll_orig_inverse_batch(image):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_ll_orig_inverse(image[i,:,:,:]))
    return torch.from_numpy(np.stack(tmp1,axis=0)) 

def get_HFmasked_ll(image,keep=0.1,lev=1,wvlt='db1'):
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeffs[1] = tuple([np.zeros(list(coeffs[1])[0].shape)]*3) 
        tmp.append(pywt.waverec2(coeffs[:2], wavelet=wvlt))
    return np.stack(tmp, axis=0)



def get_HFmasked_ll_batch(image,lev=1):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFmasked_ll(image[i,:,:,:],lev=lev))
    return torch.from_numpy(np.stack(tmp1,axis=0)) 

def get_HFtop_comp(image,keep=0.1,lev=1,wvlt='db1'):
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr*ind
        coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        tmp.append(pywt.waverec2(coeffs_filt[:2], wavelet=wvlt))
    return tmp

def get_HFtop_compfrom_HF(image,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1

    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
#         coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new = []
        for i in range(3):
            ind = np.abs(coeffs[1][i]) > thresh
            coeffs_new.append(coeffs[1][i]*ind)
#         coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp

def get_HFtop_compfrom_HF_batch(image,keep=0.1,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(np.stack(get_HFtop_compfrom_HF(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt),axis=0))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def get_HFtop_compfrom_HF_inv(image,pred_ll1,keep=0.1,lev=1,wvlt='db1'):
    if lev != 1 :
        print('function only for level 1')
        exit()
    image = image.detach().cpu()
    pred_ll1 = pred_ll1.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
#         print(coeffs_HF.shape)           #768
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new.append(coeffs[1][j]*ind)
#         print(pred_ll1[i,:,:].shape)
        tmp.append(pywt.waverec2([pred_ll1[i,:,:], tuple(coeffs_new)], wavelet=wvlt))
    return tmp

def get_HFtop_compfrom_HF_inv_batch(image,pred_ll1,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFtop_compfrom_HF_inv(image[i,:,:,:],pred_ll1[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt))
    return torch.from_numpy(np.stack(tmp1,axis=0))


def get_HFtop_comp_lev(image,keep=0.15,lev=1,wvlt='db1'):
    if lev != 1 :
        print('function only for level 1')
        exit()
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        coeff_arr_new = coeff_arr[:16,:16]
        Csort = np.sort(np.abs(coeff_arr_new.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr_new) > thresh
        coeff_arr[:16,:16] = coeff_arr_new*ind
        coeffs_filt = pywt.array_to_coeffs (coeff_arr,coeff_slices, output_format='wavedec2')
        tmp.append(pywt.waverec2(coeffs_filt[:2], wavelet=wvlt))
    return tmp

def get_HFtop_comp_lev_batch(image,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(np.stack(get_HFtop_comp_lev(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt),axis=0))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def get_HFtop_comp_lev_ind(image,keep=0.1,lev=1,wvlt='db1'):
    if lev != 1 :
        print('function only for level 1')
        exit()
    image = image.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeffs_new = []
        for i in range(len(coeffs[1])):
            coeff_arr_new = coeffs[1][i]
            Csort = np.sort(np.abs(coeff_arr_new.reshape (-1)))
            thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
            ind = np.abs(coeff_arr_new) > thresh        
            coeffs_new.append(coeff_arr_new*ind)
        coeffs_new = [coeffs[0], coeffs_new]
        tmp.append(pywt.waverec2(coeffs_new, wavelet=wvlt))
    return tmp

def get_HFtop_comp_lev_ind_batch(image,keep=0.1,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFtop_comp_lev_ind(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def get_HFtop_comp_lev_inv(image,pred_ll1,keep=0.1,lev=1,wvlt='db1'):
    if lev != 1 :
        print('function only for level 1')
        exit()
    image = image.detach().cpu()
    pred_ll1 = pred_ll1.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs1 = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs1)
        coeff_arr[:16,:16] = pred_ll1[i,:,:]
        Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr*ind
        coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        tmp.append(pywt.waverec2(coeffs_filt, wavelet=wvlt))
    return tmp

def get_HFtop_comp_lev_inv_batch(image,pred_ll1,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFtop_comp_lev_inv(image[i,:,:,:],pred_ll1[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def get_HFtop_comp_inv(image,pred_ll1,keep=0.1,lev=1,wvlt='db1'):
    if lev != 1 :
        print('function only for level 1')
        exit()
    image = image.detach().cpu()
    pred_ll1 = pred_ll1.detach().cpu()
    tmp = []
    for i in range(3): 
        img = image[i,:,:]
        coeffs1 = pywt.wavedec2(img,wavelet=wvlt, level=lev+1)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs1)
        Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr*ind
        coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
        tmp.append(pywt.waverec2([pred_ll1[i,:,:], coeffs_filt[2]], wavelet=wvlt))
    return tmp

def get_HFtop_comp_inv_batch(image,pred_ll1,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFtop_comp_inv(image[i,:,:,:],pred_ll1[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def coeffs_to_array_torch(Yl,Yh):
    t = Yl
    for i in range(len(Yh)-1,-1,-1):
        t = torch.cat([torch.cat([t, Yh[i][:,:,1,:,:]], dim=3), torch.cat([Yh[i][:,:,0,:,:], Yh[i][:,:,2,:,:]], dim=3)],dim=2)
    return t

def array_to_coeffs_torch(coeff,lev):
    size = coeff.size()[-1]
    new_coeff = []
    for i in range(lev) :
        Yl, Yh = coeff[:,:,:size//2,:size//2], [coeff[:,:,size//2:,:size//2], coeff[:,:,:size//2, size//2:], coeff[:,:,size//2:,size//2:]]
        coeff = Yl
        new_coeff.append(torch.stack(Yh,dim=2))
        size = coeff.size()[-1]
    return [Yl, new_coeff]    

def get_HFtop_comp_inv_torch(args,image,pred_ll1,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1
    
    image = image.unsqueeze(0)
    pred_ll1 = pred_ll1.unsqueeze(0)
    tmp = []
    for i in range(3): 
        img = image[:,i,:,:].unsqueeze(0)
        pred = pred_ll1[:,i,:,:].unsqueeze(0)
        xfm = DWTForward(J=lev, mode='zero', wave='db1').to(args.device)
        ifm = DWTInverse(mode='zero', wave='db1').to(args.device)
        Yl, Yh = xfm(img)
        coeff_array = coeffs_to_array_torch(Yl,Yh)
        Csort,_ = torch.sort(torch.abs(coeff_array.reshape(-1)))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        ind = torch.abs(coeff_array) > thresh
        Cfilt = coeff_array*ind
        coeffs_filt = array_to_coeffs_torch (Cfilt,lev=lev)
        coeffs = coeffs_filt[1][-1]
        tmp.append(ifm((pred, [coeffs])))
    return torch.cat(tmp,axis=1)

def get_HFtop_comp_inv_torch_batch(args,image,pred_ll1,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(get_HFtop_comp_inv_torch(args,image[i,:,:,:],pred_ll1[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt))
    return torch.cat(tmp1,axis=0)

def get_HFtop_comp_batch(image,keep=0.15,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        tmp1.append(np.stack(get_HFtop_comp(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt),axis=0))
    return torch.from_numpy(np.stack(tmp1,axis=0))

def get_LL_batch(image, level=1):
    tmp1 = np.zeros(image.size())
    n = image.size()[0]
    for i in range(n):
        if level==1:
            tmp1[i,:,:,:] = get_LL(image[i,:,:,:])
        else:
            tmp1[i,:,:,:] = get_LL_l2(image[i,:,:,:],lev=level)
    return torch.from_numpy(tmp1) 

def get_LL_LH_HL_HH_batch_mnist(image, level=1):
    assert level==1
    tmp1, tmp2, tmp3, tmp4 = np.zeros(image.size()), np.zeros(image.size()), np.zeros(image.size()), np.zeros(image.size())
    n = image.size()[0]
    for i in range(n):
        if level==1:
            tmp1[i,:,:,:], tmp2[i,:,:,:], tmp3[i,:,:,:], tmp4[i,:,:,:] = get_LL_LH_HL_HH_mnist(image[i,:,:,:])
    return torch.from_numpy(tmp1), torch.from_numpy(tmp2), torch.from_numpy(tmp3), torch.from_numpy(tmp4) 

def get_LL_LH_HL_HH_mnist(image):
    image = image.detach().cpu()
    tmpA, tmpH, tmpV, tmpD = np.zeros(image.shape), np.zeros(image.shape), np.zeros(image.shape), np.zeros(image.shape)
    for i in range(1):
        img = image[i,:,:]
        coeff = pywt.dwt2(img, 'db1')
        cA, (cH, cV, cD) = coeff
        cH_new = np.zeros(cH.shape)
        cV_new = np.zeros(cV.shape)
        cD_new = np.zeros(cD.shape)
        cA_new = np.zeros(cA.shape)
        coeff_A = cA, (cH_new, cV_new, cD_new)
        coeff_H = cA_new, (cH, cV_new, cD_new)
        coeff_V = cA_new, (cH_new, cV, cD_new)
        coeff_D = cA_new, (cH_new, cV_new, cD)
        inverse_img_A = pywt.idwt2(coeff_A, 'db1')
        inverse_img_H = pywt.idwt2(coeff_H, 'db1')
        inverse_img_V = pywt.idwt2(coeff_V, 'db1')
        inverse_img_D = pywt.idwt2(coeff_D, 'db1')
        tmpA[i,:,:] = inverse_img_A
        tmpH[i,:,:] = inverse_img_H
        tmpV[i,:,:] = inverse_img_V
        tmpD[i,:,:] = inverse_img_D

    return tmpA, tmpH, tmpV, tmpD

def get_LL_batch_mnist(image, level=1):
    tmp1 = np.zeros(image.size())
    n = image.size()[0]
    for i in range(n):
        if level==1:
            tmp1[i,:,:,:] = get_LL_mnist(image[i,:,:,:])
        else:
            tmp1[i,:,:,:] = get_LL_l2_mnist(image[i,:,:,:],lev=level)
    return torch.from_numpy(tmp1) 

def get_LL_mnist(image):
#     print(image.size())
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(1):
        img = image[i,:,:]
        coeff = pywt.dwt2(img, 'db1')
        cA, (cH, cV, cD) = coeff
        cH = np.zeros(cH.shape)
        cV = np.zeros(cV.shape)
        cD = np.zeros(cD.shape)
        coeff = cA, (cH, cV, cD)
        inverse_img = pywt.idwt2(coeff, 'db1')
        tmp[i,:,:] = inverse_img
#     print(tmp.shape)
    return tmp

def get_LL_l2_mnist(image, lev=2):
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(1):
        img = image[i,:,:]
        coeff = pywt.wavedec2(img, 'db1', level=lev)
        coeffs_new = [coeff[0]]
        for j in range(1,len(coeff)):
            coeffs_new.append(tuple([np.zeros(coeff[j][0].shape)]*3))
        img_new = pywt.waverec2(coeffs_new, wavelet='db1')
        tmp[i,:,:] = img_new
    return tmp


def get_LL(image):
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(3):
        img = image[i,:,:]
        coeff = pywt.dwt2(img, 'db1')
        cA, (cH, cV, cD) = coeff
        cH = np.zeros(cH.shape)
        cV = np.zeros(cV.shape)
        cD = np.zeros(cD.shape)
        coeff = cA, (cH, cV, cD)
        inverse_img = pywt.idwt2(coeff, 'db1')
        tmp[i,:,:] = inverse_img
    return tmp

def get_LL_l2(image, lev=2):
    image = image.detach().cpu()
    tmp = np.zeros(image.shape)
    for i in range(3):
        img = image[i,:,:]
        coeff = pywt.wavedec2(img, 'db1', level=lev)
        coeffs_new = [coeff[0]]
        for j in range(1,len(coeff)):
            coeffs_new.append(tuple([np.zeros(coeff[j][0].shape)]*3))
        img_new = pywt.waverec2(coeffs_new, wavelet='db1')
        tmp[i,:,:] = img_new
    return tmp

def get_HFtop_compfrom_HF_mnist(image,keep=0.1,lev=1,wvlt='db1'):
    assert lev==1

    image = image.detach().cpu()
    tmp = []
    for i in range(1): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=1)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new.append(coeffs[1][j]*ind)
        tmp.append(pywt.waverec2([coeffs[0], tuple(coeffs_new)], wavelet=wvlt))
    return tmp

def get_HFtop_compfrom_HF_l2_mnist(image,keep=0.1,lev=1,wvlt='db1'):
    image = image.detach().cpu()
    tmp = []
    for i in range(1): 
        img = image[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
        coeffs_HF = np.array(list(coeffs[1][0].reshape(-1))+list(coeffs[1][1].reshape(-1))+list(coeffs[1][2].reshape(-1)))
        Csort = np.sort(np.abs(coeffs_HF))
        thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
        coeffs_new = []
        for j in range(3):
            ind = np.abs(coeffs[1][j]) > thresh
            coeffs_new.append(coeffs[1][j]*ind)
        coeff = [coeffs[0], tuple(coeffs_new)]    
        for j in range(2,len(coeffs)):
            coeff.append(tuple([np.zeros(coeffs[j][0].shape)]*3))
        tmp.append(pywt.waverec2(coeff, wavelet=wvlt))
    return tmp

def get_HFtop_compfrom_HF_batch_mnist(image,keep=0.1,lev=1,wvlt='db1'):
    tmp1 = []
    n = image.size()[0]
    for i in range(n):
        if lev==1:
            tmp1.append(np.stack(get_HFtop_compfrom_HF_mnist(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt),axis=0))
        else :
            tmp1.append(np.stack(get_HFtop_compfrom_HF_l2_mnist(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt),axis=0))
    return torch.from_numpy(np.stack(tmp1,axis=0))


# def get_LL_l2(image):
#     image = image.detach().cpu()
#     tmp = np.zeros(image.shape)
#     for i in range(3):
#         img = image[i,:,:]
#         coeff = pywt.wavedec2(img, 'db1', level=2)
#         tmp1 = tuple([np.zeros((8,8))]*3)
#         tmp2 = tuple([np.zeros((16,16))]*3)
#         coeffs_new = [coeff[0],tmp1,tmp2]
#         img_new = pywt.waverec2(coeffs_new, wavelet='db1')
#         tmp[i,:,:] = img_new
#     return tmp

def cosim_loss(x,y):
    return 1 -(F.cosine_similarity(x,y)).mean() 


# class LPDataset(torch.utils.data.Dataset):
#     """Dataset wrapper to induce class-imbalance"""

#     def __init__(self, dataset, radius=4):

#         self.dataset = dataset ## Recieve (Transformed) dataset
#         self.radius = radius
#         self.get_lp()

#     def get_lp(self):

#         self.lp = [None]*(len(self.dataset))
#         for idx in tqdm.tqdm(range(len(self.dataset)), total=len(self.dataset), leave=False):
#             x,_ = self.dataset[idx]
#             lp_x, _ = get_freq(x.unsqueeze(0), self.radius)
#             self.lp[idx] = lp_x.squeeze(0)

#     def __getitem__(self, i):
#         x, y = self.dataset[i] ## Original Sample
#         lp_x = self.lp[i]

#         return (x, lp_x, y)

#     def __len__(self):
#         return len(self.dataset)

class LPDataset(torch.utils.data.Dataset):
    """Dataset wrapper to induce class-imbalance"""

    def __init__(self, dataset, pre='comp_wv', radius=4, keep=0.1, level=3, wavelet='db1', path=None,test=False, affine_transform=False):

        self.dataset = dataset ## Recieve (Transformed) dataset
        self.radius = radius
        self.pre = pre
        self.keep = keep
        self.level = level
        self.wavelet = wavelet
        self.affine_transform = affine_transform
        if self.pre=='trans':
            if test==True:
#                 self.translated_data = torch.load(f'./data/translated_{path}_test.pt',map_location='cpu')
#                 self.GT_data = torch.load(f'./data/GT_{path}_test.pt',map_location='cpu')
                data = torch.load(f'./data/baseline_resnet18/{path}_test.pt',map_location='cpu')
                self.translated_data = data['Adv_trans_ll']
                self.GT_data = data['Adv_Orig']
            else:
#                 self.translated_data = torch.load(f'./data/translated_{path}.pt', map_location='cpu')
#                 self.GT_data = torch.load(f'./data/GT_{path}.pt',map_location='cpu')
                data = torch.load(f'./data/baseline_resnet18/{path}_train.pt',map_location='cpu')
                self.translated_data = data['Clean_trans_ll']
                self.GT_data = data['Clean_Orig']
        if self.pre=='Adv':
                self.adv_data = torch.load(f'./data/baseline_resnet18/Cifar10_PGD_step7_train.pt',map_location='cpu')
#                 self.orig_data = torch.load(f'./data/baseline_resnet18/orig_data_train.pt',map_location='cpu')

        self.get_lp()
        
    def transform(self,img1,img2):
        img1 = transforms.Pad(padding=4)(img1)
        img2 = transforms.Pad(padding=4)(img2)
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(32,32))
        img1 = torchvision.transforms.functional.crop(img1, i, j, h, w)
        img2 = torchvision.transforms.functional.crop(img2, i, j, h, w)
        if random.random() > 0.5:
            img1 = torchvision.transforms.functional.hflip(img1)
            img2 = torchvision.transforms.functional.hflip(img2)
        return img1, img2  
    
    def transform_single(self,img1):
        img1 = transforms.Pad(padding=4)(img1)
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(32,32))
        img1 = torchvision.transforms.functional.crop(img1, i, j, h, w)
        if random.random() > 0.5:
            img1 = torchvision.transforms.functional.hflip(img1)
        return img1 

    def get_lp(self):

        self.lp = [None]*(len(self.dataset))
        self.gt = [None]*(len(self.dataset))

        for idx in tqdm.tqdm(range(len(self.dataset)), total=len(self.dataset), leave=False):
            x,_ = self.dataset[idx]
            if self.pre=='lp' :    
                x = x.unsqueeze(0)
                lp_x, _ = get_freq(x, self.radius)
                lp_x = lp_x.squeeze(0)
                self.lp[idx] = lp_x
            elif self.pre=='comp_wv' :
                lp_x = get_comp_wv(x,keep=self.keep,lev=self.level,wvlt=self.wavelet)
                self.lp[idx] = torch.from_numpy(lp_x)
            elif self.pre=='trans' :
                self.lp[idx] = self.translated_data[idx].detach().cpu()
                self.gt[idx] = self.GT_data[idx].detach().cpu()
            elif self.pre=='Adv' :
#                 print(idx)
                self.lp[idx] = self.adv_data['Adv'][idx].squeeze(0).detach().cpu()
#                 self.gt[idx] = self.orig_data[idx].squeeze(0).detach().cpu()

                

    def __getitem__(self, i):
        x, y = self.dataset[i] ## Original Sample
        lp_x = self.lp[i]
        gt_x = self.gt[i]
        if self.affine_transform==True:
#             print('Using paired transform')
            lp_x, gt_x = self.transform(lp_x, gt_x)
            
        return ( x, lp_x, y)


#     def __getitem__(self, i):
#         x, y = self.dataset[i] ## Original Sample
#         lp_x = self.lp[i]
#         lp_x = self.transform_single(lp_x)
        
#         return (x, gt_x, lp_x, y)

    def __len__(self):
        return len(self.dataset)


class IdxDataset(torch.utils.data.Dataset):
    """Dataset wrapper to induce class-imbalance"""

    def __init__(self, dataset):

        self.dataset = dataset ## Recieve (Transformed) dataset

    def __getitem__(self, i):
        x, y = self.dataset[i] ## Original Sample

        return (x, y, i)

    def __len__(self):
        return len(self.dataset)

def mixup_data(x, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
    return torch.cat([x, mixed_x],0)     #, y_a, y_b, lam 


def get_wv(data, lev=1):

    coefficients = np.zeros(data.shape)
    coefficients_slices = []

    for i in range(3): 
        img = data[i,:,:]
        coeffs = pywt.wavedec2(img,wavelet='db1', level=lev)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        coefficients_slices.append(coeff_slices)
        coefficients[i,:,:] = coeff_arr
    
    return coefficients, coefficients_slices

def recon_data(data, coeffs, slices):
    
    tmp = np.zeros(data.shape)

    for i in range(3):
        coeffs_new = pywt.array_to_coeffs(coeffs[i,:,:], slices[i], output_format='wavedec2')
        new_img_c = pywt.waverec2(coeffs_new, wavelet='db1')
        tmp[i,:,:] = new_img_c
    
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    return tmp

def mixup_wavelet(x, lev=1, use_cuda=True):

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mask_idx = x.size()[2] // (2*lev) 
    # print(mask_idx)

    for i, j in zip(range(batch_size), index):
        
        wv_1, slices = get_wv(x[i], lev=lev)
        mask_1 = np.zeros(x[0].shape)
        mask_1[:, :mask_idx, :mask_idx] = 1
        wv_1_masked = wv_1 * mask_1

        # if i==4:
        #     plt.imshow(np.transpose(wv_1_masked, (1,2,0)));

        wv_2, _ = get_wv(x[j.item()], lev=lev)
        mask_2 = 1 - mask_1
        wv_2_masked = wv_2 * mask_2

        wv_new = wv_1_masked + wv_2_masked
        img_new = recon_data(x[0], wv_new, slices)

        # print(np.min(img_new), np.max(img_new))
        # plt.imshow(np.transpose(img_new, (1,2,0)))

        img_new_tensor = torch.from_numpy(img_new).unsqueeze(0)
        # print(type(img_new_tensor), img_new_tensor.shape)

        x = torch.cat((x, img_new_tensor), 0)
    
    # print(type(img_cat), img_cat.shape)
    return x

def mixup_wavelet_custom(x, lev=1, use_cuda=True):

    assert lev==2

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    # print(index)

    mask_idx = x.size()[2] // (2*lev) 
    # print(mask_idx)

    for i, j in zip(range(batch_size), index):
        
        wv_1, slices = get_wv(x[i], lev=lev)
        
        mask_1 = np.zeros(x[0].shape)
        mask_1[:, :mask_idx, :mask_idx] = 1
        mask_1[:, mask_idx:2*mask_idx, :mask_idx] = 1
        mask_1[:, :mask_idx, mask_idx:2*mask_idx] = 1
        wv_1_masked = wv_1 * mask_1

        # if i==4:
        #     plt.imshow(np.transpose(wv_1_masked, (1,2,0)));

        wv_2, _ = get_wv(x[j.item()], lev=lev)
        mask_2 = 1 - mask_1
        wv_2_masked = wv_2 * mask_2

        wv_new = wv_1_masked + wv_2_masked
        img_new = recon_data(x[0], wv_new, slices)

        # print(np.min(img_new), np.max(img_new))
        # plt.imshow(np.transpose(img_new, (1,2,0)))

        img_new_tensor = torch.from_numpy(img_new).unsqueeze(0)
        # print(type(img_new_tensor), img_new_tensor.shape)

        x = torch.cat((x, img_new_tensor), 0)
    
    # print(type(img_cat), img_cat.shape)
    return x

def gaussian_noise_cat(x, mean=0., stddev=0.25):
    x_pert = x + torch.randn(x.size())*stddev + mean
    return torch.cat((x, x_pert), 0)

def gaussian_noise(x, mean=0., stddev=0.25):
    x_pert = x + torch.randn(x.size())*stddev + mean
    return x_pert

def load_data(batch_size=128,transform_type=None,radius=6,return_data=False,translated_path=None, idx_dataset=False, transform=False , use_synthetic_dataset=False, synth_root=None, adv_attack=None, dataroot="./datasets/cifar_10"):

    print('==> Preparing data..')

    transform_list = []
    if transform==True:
        print('Using Transform')
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if not use_synthetic_dataset:
        transform_list.append(transforms.ToTensor())
    
    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([ transforms.ToTensor(), transforms.Resize((32,32))])

    if use_synthetic_dataset:  # synthetic dataset is in tensor fomat. synth_root is root location of dataset,adv_attack is not None if we want to load adversarial image
        print("using synthetic training dataset")
        trainset = SyntheticDataset(synth_root, adv_attack, transform_train)
    else:
        if "cifar_10" in dataroot:
            trainset = torchvision.datasets.CIFAR10(
                root=dataroot, train=True, download=True, transform=transform_train)
        
        elif "fmnist" in dataroot:
            trainset= torchvision.datasets.FashionMNIST(
                root = dataroot,train = True,download = True,transform = transform_train
            )
        elif "mnist" in dataroot:
            trainset= torchvision.datasets.MNIST(
                root = dataroot,train = True,download = True,transform = transform_train
            )
        elif "svhn" in dataroot:
            trainset= torchvision.datasets.SVHN(
            root = dataroot,split="train",download = True,transform = transform_test
        )
        else:
            raise(Exception("unknown train dataroot { }".format(dataroot)))
    

    if "cifar_10" in dataroot:
        testset = torchvision.datasets.CIFAR10(
                root=dataroot, train=False, download=True, transform=transform_test)

    elif "fmnist" in dataroot:
        testset= torchvision.datasets.FashionMNIST(
            root = dataroot,train = False,download = True,transform = transform_test
        )
    elif "mnist" in dataroot:
        testset= torchvision.datasets.MNIST(
            root = dataroot,train = False,download = True,transform = transform_test
        )
    elif "svhn" in dataroot:
        testset= torchvision.datasets.SVHN(
            root = dataroot,split="test",download = True,transform = transform_test
        )
    else:
        raise(Exception("unknown dataroot { }".format(dataroot)))
    
    # train_size = 2000
    # test_size = 48000
    # trainset, test_dataset = torch.utils.data.random_split(trainset, [train_size, test_size])        

    
    if return_data:
        print('==> Using Dataset')
        return trainset, testset
    if transform_type!=None:
        print(f'Loading Test Data, {transform_type}, {translated_path}')
        # trainset = LPDataset(trainset,radius=radius,pre=transform_type,path=translated_path, affine_transform=transform)  
        # LPDataset stores low pass images
        testset = LPDataset(testset,radius=radius,pre=transform_type,path=translated_path,test=True)  
    
    if idx_dataset:
        trainset = IdxDataset(trainset) 
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def load_synthetic_train_dataset(synth_root, adv_attack, batch_size=64):

    transform_list=[]
    transform_train = transforms.Compose(transform_list)
    trainset = SyntheticDataset(synth_root, adv_attack, None)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    return trainloader

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std

        if input.size(1)==1 and len(self.mean) > 1:
                mean=mean[:,0,:,:].unsqueeze(1)
                std = std[:,0,:,:].unsqueeze(1)

        return (input - mean) / std
    
class Denormalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Denormalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std

        if input.size(1)==1 and len(self.mean) > 1:
                mean=mean[:,0,:,:].unsqueeze(1)
                std = std[:,0,:,:].unsqueeze(1)

        return (input*std)+mean


class MinMaxNormalization(nn.Module) :
    def __init__(self) :
        super(MinMaxNormalization, self).__init__()
#         self.register_buffer('mean', torch.Tensor(mean))
#         self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        
        return (input - torch.min(input)) / (torch.max(input) - torch.min(input))


def load_model(args, load_as_D=False):
    if args.teacher_model=='resnet18':
        base_model = ResNet18()
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model = nn.Sequential(norm_layer,base_model).to(args.device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if args.load_path is not None:
            state_dict = torch.load(args.load_path, map_location=args.device)
            print(state_dict.keys())
            print(f'Loading Model : Best Acc : {state_dict["acc"]} \t|\t Epoch : {state_dict["epoch"]}')
    
            model.load_state_dict(state_dict["net"]) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(args.device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)
        
                
    elif args.teacher_model=='vgg':
        base_model = VGG('VGG11',channels=1)
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model = nn.Sequential(norm_layer,base_model).to(args.device)

        if args.load_path is not None :
            state_dict = torch.load(args.load_path, map_location=args.device)
            print(f'Loading Model : Best Acc : {state_dict["acc"]} \t|\t Epoch : {state_dict["epoch"]}')
            model.load_state_dict(state_dict["net"])      
            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(args.device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)
                
                
    elif args.teacher_model=='wrn':
        _net_builder = build_WideResNet(depth=28, widen_factor=2, leaky_slope=0.1, dropRate=0.0, use_embed=False)
        base_model = _net_builder.build(10)
        checkpoint = torch.load(args.load_path)
        load_model = checkpoint['ema_model']
        new_state_dict = OrderedDict()
        for k, v in load_model.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v    
        base_model.load_state_dict(new_state_dict)
        norm_layer = Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]])
        model = nn.Sequential(norm_layer,base_model).to(args.device)
      
    return model


# while training --surrogate_teacher_model argument is for teacher model used for training_file
#but while testing --surrogate_teacher_model argument is for  model on which testing is done. It could be the model used for training or any other model

def load_model_1(model_name, model_path, device,load_as_D=False , input_channels=3):
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        if "net" in state_dict:
            state_dict = state_dict["net"]

    if model_name=="resnet18":
        base_model = ResNet18()
        
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
            
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)

    
    if model_name=="resnet18_hard_label":
        base_model = ResNet18()
        
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
            
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)

    #TODO convert duplicated code in seperate function
    if model_name=="half_alexnet":
        base_model = HalfAlexnet(input_channels=input_channels)
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   #as per black_box_ripper code this mean and std was used
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
           
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)
    
    if model_name=="resnet34":
        base_model = ResNet34()
        
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
            
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)
    
    if model_name=="alexnet":
        base_model = Alexnet(input_channels=input_channels)
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   #as per black_box_ripper code this mean and std was used
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
           
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)

    if model_name=="resnet18_black_box":  # architecture of resnet18 from black box ripper is little diffrent
        base_model = BlackBoxResNet18(input_channels=input_channels)
        
        norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(norm_layer,base_model).to(device)

        #surrogate models pth file is differnt from other models. we need to load it seperately
        if model_path is not None:
            
            base_model.load_state_dict(state_dict) 

            if load_as_D==True :
                print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
                Classifier = nn.Linear(512*model[1].expansion, 1).to(device)
                print('Teacher Classifier :', model[1].linear)
                model[1].linear = Classifier
                print('Linear Classifier :', model[1].linear)

    return model
# def load_model(args, load_as_D=False):
#     if args.teacher_model=='resnet18':
#         base_model = ResNet18()
#         norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#         model = nn.Sequential(norm_layer,base_model).to(args.device)
# #         print(norm_layer.state_dict())
# #         print(model.parameters())
# #         for param_tensor in norm_layer.state_dict():
# #             print(param_tensor, "\t", norm_layer.state_dict()[param_tensor])
# # #         print(model[0])
# #         exit()

#         if args.load_path is not None:
#             state_dict = torch.load(args.load_path, map_location=args.device)
#             state_dict["state_dict"]["0.mean"] = model.state_dict()["0.mean"]
#             state_dict["state_dict"]["0.std"] = model.state_dict()["0.std"]
# #             print(state_dict["state_dict"].keys())
# #             exit()
# #             print(f'Loading Model : Best Acc : {state_dict["acc"]} \t|\t Epoch : {state_dict["epoch"]}')
# #             base_model = nn.Sequential(base_model).to(args.device)
#             model.load_state_dict(state_dict["state_dict"])   
# #             model = nn.Sequential(norm_layer,base_model).to(args.device)
    
#             if load_as_D==True :
#                 print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
#                 Classifier = nn.Linear(512*model[1].expansion, 1).to(args.device)
#                 print('Teacher Classifier :', model[1].linear)
#                 model[1].linear = Classifier
#                 print('Linear Classifier :', model[1].linear)
                
#     elif args.teacher_model=='vgg':
#         base_model = VGG('VGG16',channels=1)
#         norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         model = nn.Sequential(norm_layer,base_model).to(args.device)

#         if args.load_path is not None:
#             state_dict = torch.load(args.load_path, map_location=args.device)
#             print(f'Loading Model : Best Acc : {state_dict["acc"]} \t|\t Epoch : {state_dict["epoch"]}')
#             model.load_state_dict(state_dict["net"])      
#             if load_as_D==True :
#                 print("Using Teacher as Discriminator by replacing the last layer by binary classifier.................")
#                 Classifier = nn.Linear(512*model[1].expansion, 1).to(args.device)
#                 print('Teacher Classifier :', model[1].linear)
#                 model[1].linear = Classifier
#                 print('Linear Classifier :', model[1].linear)
                
                
#     elif args.teacher_model=='wrn':
#         _net_builder = build_WideResNet(depth=28, widen_factor=2, leaky_slope=0.1, dropRate=0.0, use_embed=False)
#         base_model = _net_builder.build(10)
#         checkpoint = torch.load(args.load_path)
#         load_model = checkpoint['ema_model']
#         new_state_dict = OrderedDict()
#         for k, v in load_model.items():
#             name = k[7:] # remove `module.`
#             new_state_dict[name] = v    
#         base_model.load_state_dict(new_state_dict)
#         norm_layer = Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]])
#         model = nn.Sequential(norm_layer,base_model).to(args.device)
        
#     return model


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)        
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()      
    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, model,args):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.resize = resize
#         self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)).to(args.device)
#         self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)).to(args.device)
        self.norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(args.device)

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
#         input = (input-self.mean) / self.std
#         target = (target-self.mean) / self.std
        input = self.norm_layer(input)
        target = self.norm_layer(target)
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
#             if i in style_layers:
#                 act_x = x.reshape(x.shape[0], x.shape[1], -1)
#                 act_y = y.reshape(y.shape[0], y.shape[1], -1)
#                 gram_x = act_x @ act_x.permute(0, 2, 1)
#                 gram_y = act_y @ act_y.permute(0, 2, 1)
#                 loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class Res18PerceptualLoss(torch.nn.Module):
    def __init__(self, model,args):
        super(Res18PerceptualLoss, self).__init__()
        self.conv1 = model.conv1.eval()
        self.bn1 = model.bn1.eval()
        blocks = []
        blocks.append(model.layer1.eval())
        blocks.append(model.layer2.eval())
        blocks.append(model.layer3.eval())
        blocks.append(model.layer4.eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.resize = resize
#         self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)).to(args.device)
#         self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)).to(args.device)
        self.norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(args.device)

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
#         input = (input-self.mean) / self.std
#         target = (target-self.mean) / self.std
        input = self.norm_layer(input)
        target = self.norm_layer(target)
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = F.relu(self.bn1(self.conv1(input)))
        y = F.relu(self.bn1(self.conv1(target)))
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += cosim_loss(x, y)
#             if i in style_layers:
#                 act_x = x.reshape(x.shape[0], x.shape[1], -1)
#                 act_y = y.reshape(y.shape[0], y.shape[1], -1)
#                 gram_x = act_x @ act_x.permute(0, 2, 1)
#                 gram_y = act_y @ act_y.permute(0, 2, 1)
#                 loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def TV_loss(y_hat):
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    return diff_i+diff_j

class spectral_loss():
    def __init__(self,args):
        self.xfm = DWTForward(J=1, mode='zero', wave='db1').to(args.device)
        self.L2loss = nn.MSELoss(reduction='sum')
    def loss(self,Y_hat,Y):
        Yl, Yh = self.xfm(Y)
        Yhatl, Yhath = self.xfm(Y_hat)
        loss = self.L2loss(Yhatl, Yl) + self.L2loss(Yhath[0], Yh[0])
#         print(self.L2loss(Yhath[0], Yh[0]))
        return loss

def visualize_inputs(image, path):
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    plt.imshow(image.detach().cpu().numpy())
    plt.savefig(path)    



# def get_comp_wv_lev1(image,keep=0.1,lev=3,wvlt='db1'):
#     image = image.detach().cpu()
#     tmp = np.zeros((image.size()[0],16,16))
#     tmp2 = np.zeros((image.size()[0],16,16))
#     for i in range(3): 
#         img = image[i,:,:]
#         coeffs = pywt.wavedec2(img,wavelet=wvlt, level=lev)
#         coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
#         Csort = np.sort(np.abs(coeff_arr.reshape (-1)))
#         thresh = Csort[int(np.floor((1-keep) *len (Csort)))]
#         ind = np.abs(coeff_arr) > thresh
#         Cfilt = coeff_arr*ind
#         coeffs_filt = pywt.array_to_coeffs (Cfilt,coeff_slices, output_format='wavedec2')
#         Arecon = pywt.waverec2(coeffs_filt[:-1], wavelet=wvlt)
#         Arecon2 = pywt.waverec2(coeffs[:-1], wavelet=wvlt)
#         tmp[i,:,:] = Arecon
#         tmp2[i,:,:] = Arecon2
#     return tmp, tmp2

# def get_comp_wv_batch_lev1(image,keep=0.1,lev=3,wvlt='db1'):
#     tmp1 = np.zeros((image.size()[0],image.size()[1],16,16))
#     tmp2 = np.zeros((image.size()[0],image.size()[1],16,16))
#     n = image.size()[0]
#     for i in range(n):
#         tmp1[i,:,:,:], tmp2[i,:,:,:] = get_comp_wv_lev1(image[i,:,:,:],keep=keep,lev=lev,wvlt=wvlt)
#     return torch.from_numpy(tmp1), torch.from_numpy(tmp2) 



def prog_k(curr_epoch, epoch_min=1, epoch_max=200, k_min=15, k_max=90):
    ## y = (y2-y1)/(x2-x1) * (x - x1) + y1
    k = (k_max - k_min) / (epoch_max - epoch_min) * (curr_epoch - epoch_min) + k_min
    return 100 - int(k)


import os.path
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self,root , attack=None, transform=None):
        
        """
        :param root:  root directory
        :param batch_size: number of images in batch file
        """
        
        self.root = root
        self.images = torch.load(os.path.join(root, "images.pt")).detach()
        self.labels = torch.load(os.path.join(root, "labels.pt")).detach()
        self.labels=self.labels.type(torch.LongTensor)

        assert torch.min(self.images) >= 0 and torch.max(self.images) <=1
        assert torch.min(self.labels) >= 0 and torch.max(self.labels) <=9

        self.size = self.labels.size(0)
        self.adv=None
        if os.path.exists(os.path.join(root, "adv_images.pt")):
            self.adv= torch.load(os.path.join(root, "adv_images.pt")).detach()
        else:
            if attack is not None:
                i=0
                step=128
                while i< self.size:
                    e = min(i+step, self.size)
                    adv = (attack(self.images[i:e], self.labels[i:e])).cpu()
                    i+=step
                    if self.adv is None:
                        self.adv = adv
                    else:
                        self.adv = torch.cat((self.adv, adv), 0)
            
            torch.save(self.adv, os.path.join(root, "adv_images.pt"))
       

        self.transform = transform


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        if self.adv is not None and len(self.adv) >0   :
            adv = self.adv[idx]
            if self.transform is not None:
                adv = self.transform(adv)
        else:
            adv = None
        
        image = self.images[idx]
        
        if self.transform is not None:
            image=  self.transform(image)


        if self.adv is not None and len(self.adv) >0   :
            """if idx%1000==0:
                torchvision.utils.save_image(adv[0] , "tobedeleted/adv_{}.png".format(idx))
                torchvision.utils.save_image(image[0] , "tobedeleted/img_{}.png".format(idx))"""
            return  image  , adv ,  self.labels[idx]
        else:
            return image[0,:,:].unsqueeze(0), self.labels[idx]

def wvlt_transform(images, device, lev, selection_strategy,wvlt='db1', mode="symmetric", keep=None,setting=None, threshold_strategy="soft"):
    """
        keep : ratio of coeefficients to keep
        lev : decomposition level
        setting : which coeefficients to take, top, bottom or random k% of coeefficients
        select_from_all : if set to True , select k% coefficients from all coeefficients. Otherwise select all coefficients from Yl and select k% from Yh
    """
    
    xfm = DWTForward(J=lev, mode=mode, wave=wvlt).to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode=mode, wave=wvlt).to(device)  # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(images)
    batch_size = images.size(0)
    channels = images.size(1)
    
    if selection_strategy =="ll":
        for l in range(lev):
            Yh[l]*=0
        
        
    elif selection_strategy =="wv_custom":
        temp = Yh[0].view(batch_size, channels, -1)
        for l in range(1, lev):
            temp = torch.cat((temp, Yh[l].view(batch_size, channels, -1)), dim=2)
        
        Csort, _ = torch.sort(torch.abs(temp), dim=2)
        s = Csort.size(2)
        if setting=="top":
            thresh = Csort[:, :, int(np.floor((1 - keep) * s))].view(batch_size, channels, 1, 1, 1)
            for l in range(0, lev):
                ind = torch.abs(Yh[l]) > thresh
                Yh[l] *= ind
        elif setting=="bottom":
            thresh = Csort[:, :, int(np.floor((keep) * s))].view(batch_size, channels, 1, 1, 1)
            for l in range(0, lev):
                ind = torch.abs(Yh[l]) < thresh
                Yh[l] *= ind
        elif setting =="random":
            for l in range(lev):
                ind  = torch.randn_like(Yh[l])
                ind = F.dropout(ind, p=(1-keep))
                ind = torch.where(ind!=0, 1, 0)
                Yh[l]*=ind
        else:
            raise Exception("unknown setting")
    elif selection_strategy=="visushrink":
        var = get_var(Yh[0])
        num_pixels = torch.tensor(images.size(2)*images.size(3) , device=torch.device(device))
        threshold = vis_shrink_2d(var, num_pixels).unsqueeze(3).unsqueeze(4)
        for l in range(0,lev):
            if threshold_strategy =="soft":
                Yh[l] = torch.where(torch.abs(Yh[l]) > threshold , torch.abs(Yh[l]) - threshold , 0)
                Yh[l] = Yh[l]*torch.sign(Yh[l]) 
            elif threshold_strategy =="hard":
                Yh[l] = torch.where(torch.abs(Yh[l]) > threshold , Yh[l] - threshold , 0)
            else:
                raise("unknown thresholding strategy")
    elif selection_strategy == "bayes":
        x = images.detach().cpu().numpy()
        x  = denoise_wavelet(x , channel_axis=1, convert2ycbcr=True,
                    method='BayesShrink', mode=threshold_strategy,
                    rescale_sigma=True)
        x = torch.from_numpy(x).to(device).contiguous()
        
				
    elif selection_strategy =="rigsure":
        var = get_var(Yh[0])    #Yh[0] is finest resolution
        for l in range(0,lev): 
            threshold = rig_sure(Yh[l],var)
            if threshold_strategy =="soft":
                sign = torch.sign(Yh[l]) 
                Yh[l] = torch.where(torch.abs(Yh[l]) > threshold , torch.abs(Yh[l]) - threshold , 0)
                Yh[l] = Yh[l]*sign
            elif threshold_strategy =="hard":
                Yh[l] = torch.where(torch.abs(Yh[l]) > threshold , Yh[l] , 0)
            else:
                raise Exception("unknown thresholding strategy")

    else:
        raise Exception("unknown selection strategy")
    images = ifm((Yl, Yh))
    return images


    

if __name__ == '__main__':
    #n = Denormalize([0.5, 0.5, 0.5] , [0.5, 0.5 , 0.5])
    x = torch.randn((128,3,32,32))
    y=x
    #y = n(x)
    device = torch.device('cuda')
    y = y.to(device)
    z = wvlt_transform(y,device,2,"wv_custom",keep=1, setting="top")
    print(format(torch.sum(torch.abs(y-z)), '.8f'))
    print(format(torch.sum(torch.abs(y)), '.8f'))
    print(format(torch.sum(torch.abs(z)), '.8f'))