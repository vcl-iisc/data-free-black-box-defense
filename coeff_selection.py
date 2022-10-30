import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import pywt
import os
import sys
import time
import math
import tqdm
import torchattacks
import random
import numpy as np
from models.resnet import ResNet18
from torch.autograd import Variable

import argparse

import torchvision.datasets as datasets
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
root='./data', train=False, download=True, transform=transform_test) 
testloader = torch.utils.data.DataLoader(
        testset, batch_size=20, shuffle=False, num_workers=2)


parser = argparse.ArgumentParser(description='Check Detector Performance')
    
parser.add_argument('--case',help='experiment-scenario',default='1')
parser.add_argument("--attack",help='Attack choice', default = "auto_attack", choices=['pgd', 'auto_attack'],type=str)
parser.add_argument('--gpu',help='Model Choice', default='0')
parser.add_argument('--score',help='which metric to use for score',default='kl')
 
args = parser.parse_args()

kl_criterion = nn.KLDivLoss(reduction='batchmean')



def load_model():
 
    base_model = ResNet18()
    norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(f'cuda:{args.gpu}')
    model = nn.Sequential(norm_layer,base_model)

    state_dict = torch.load('./checkpoint_cifar10/baseline_resnet18.pth', map_location='cpu')
    print(f'Loading Model : Best Acc : {state_dict["acc"]} \t|\t Epoch : {state_dict["epoch"]}')
    model.load_state_dict(state_dict["net"])
    model.to('cpu')     

    return model

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
        return (input - mean) / std

model = load_model()
model = model.to(f'cuda:{args.gpu}')


def get_wv(data, lev=1):

    coefficients = np.zeros(data.shape)
    coefficients_slices = []

    for i in range(3): 
        img = data[i,:,:].detach().cpu()
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


def score_metric(new_data, orig_data):

    orig_data = torch.from_numpy(orig_data).to(f'cuda:{args.gpu}' ,dtype=torch.float) #.to('cuda:0', dtype=torch.float)
    new_data = torch.from_numpy(new_data).to(f'cuda:{args.gpu}', dtype=torch.float)#.to('cuda:0', dtype=torch.float)
    
    model.eval()

    og_op = model(orig_data)
    og_score, og_idx = og_op.max(1)

    new_op = model(new_data)    
    new_score = new_op[0][og_idx.item()]

    _, new_idx = new_op.max(1) ## Argmax for new index

    T = 1
    new_p, og_p = F.log_softmax(new_op/T, dim=1), F.softmax(og_op/T, dim=1)

    ## How imp is this coeff?
    if args.score == 'kl':
        score = (T * T) * kl_criterion(new_p, og_p).item() ## SCORE using KL-Divergence    

    elif args.score == 'kl_argmax':

        if new_idx==og_idx:
            score = (T * T) * kl_criterion(new_p, og_p).item() ## SCORE using KL-Divergence
        else:
            score = 0

    else:
        score = torch.abs(new_score - og_score) ## If high-> very sensitive ; If Low -> not very sensitive
        
    return np.abs(score)



def scoring_fn(wv, data, slices):

    orig_data = recon_data(data, wv, slices)
    row, col = wv.shape[1], wv.shape[2]

    heatmap = np.zeros((row, col))

    ## For each coefficient
    for i in range(row):
        for j in range(col):

            wv_new = np.copy(wv)
            wv_new[:, i, j] = 0

            new_data = recon_data(data, wv_new, slices)

            score = score_metric(new_data, orig_data)
            heatmap[i, j] = score

    return heatmap



def get_corr(reg='LL', lev_idx = 16):

    if reg=='LL':
        hor = (0, lev_idx)
        ver = (0, lev_idx)

    elif reg=='LH':
        hor = (lev_idx, 2*lev_idx)
        ver = (0, lev_idx)

    elif reg=='HL':
        hor = (0, lev_idx)
        ver = (lev_idx, 2*lev_idx)

    elif reg=='HH':
        hor = (lev_idx, 2*lev_idx)
        ver = (lev_idx, 2*lev_idx)
    else: 
        ## Take all
        hor = (0, 2*lev_idx)
        ver = (0, 2*lev_idx)

    return hor, ver


def norm_heatmap(heatmap, norm_ll=False, inv=True):
    ## Normalize the heatmap region wise

    ## LL
    hor,ver = get_corr(reg='LL')
    x1,x2 = hor
    y1,y2 = ver

    if norm_ll:
        ll_max, ll_min = np.max(heatmap[x1:x2, y1:y2]), np.min(heatmap[x1:x2, y1:y2])

        # print(f'LL Max: {ll_max} | Min: {ll_min}')
        heatmap[x1:x2, y1:y2] = (heatmap[x1:x2, y1:y2] - ll_min) / (ll_max - ll_min)
    else:
        heatmap[x1:x2, y1:y2] = 1 ## Select everything from LL

    ## HF regions
    heatmap_no_ll = np.copy(heatmap)
    heatmap_no_ll[x1:x2, y1:y2] = 0 ## Set LL to zero
    hf_max, hf_min = np.max(heatmap_no_ll), np.min(heatmap_no_ll) ## Find max and min
    regions = ['LH', 'HL', 'HH']

    # print(f'HF Max: {hf_max} | Min: {hf_min}')

    for reg in regions:
        hor,ver = get_corr(reg=reg)
        x1,x2 = hor
        y1,y2 = ver

        heatmap[x1:x2, y1:y2] =  np.copy((heatmap_no_ll[x1:x2, y1:y2] - hf_min) / (hf_max - hf_min))

        if inv:
            heatmap[x1:x2, y1:y2] = 1 - heatmap[x1:x2, y1:y2] ## Inversion

    return heatmap


def get_norm_spatial_sample(data, idx, norm_ll=False, inv=True, only_ll=False):

    wv, slices = get_wv(data[idx])     ## Calculate Wavelet
    if only_ll:
        # print(f'Doing only LL')
        heatmap_norm = np.zeros((3,32,32))
        heatmap_norm[:, :16, :16] = 1
    
    else:
        heatmap = scoring_fn(wv, data[idx], slices) ## Calculate Heatmap
        heatmap_norm = norm_heatmap(heatmap, norm_ll=norm_ll, inv=inv) ## Calculated normalized inverted heatmap

    wv_filt = wv * heatmap_norm ## Calculate     
    new_data = recon_data(data[idx], wv_filt, slices)
    new_data = torch.from_numpy(new_data).to(f'cuda:{args.gpu}', dtype=torch.float)#.to('cuda:0', dtype=torch.float)

    return new_data


def calc_acc(loader, model, attack, adv=True, norm_ll=False, inv=True, only_ll=False):
    
    total, correct = 0,0
    model.eval()

    for batch_num, (data, labels) in enumerate(loader):

        # if batch_num == 5:
        #     print(f'Accuracy: {(correct/total)*100} at batch: {batch_num}| Correct: {correct} | Total: {total}')
        #     return correct, total
        
        batch_size = data.size(0)
        total += batch_size

        if adv:
            ## Get adv. sample
            data = attack(data, labels)

        for idx in range(batch_size):
            
            data_filt = get_norm_spatial_sample(data, idx, norm_ll=norm_ll, inv=inv, only_ll=only_ll)
            op = model(data_filt)
            _, pred = op.max(1)
            correct += sum(pred.cpu()==labels[idx])
        
        print(f'batch-{batch_num}: {(correct/total)*100} | Corr: {correct} | Total: {total}')

    print(f'Accuracy: {(correct/total)*100} | Correct: {correct} | Total: {total}')
    return correct, total

def calc_acc_full(loader, model, attack, adv=True):
    
    total, correct = 0,0
    model.eval()

    for batch_num, (data, labels) in enumerate(loader):

        if batch_num == 5:
            print(f'Accuracy: {(correct/total)*100} at batch: {batch_num}| Correct: {correct} | Total: {total}')
            return correct, total
        
        batch_size = data.size(0)
        total += batch_size

        if adv:
            ## Get adv. sample
            data = attack(data, labels)

        data = data.to(f'cuda:{args.gpu}', torch.float)

        op = model(data)
        _, pred = op.max(1)
        correct += sum(pred.cpu()==labels)
        
        print(f'batch-{batch_num}: {(correct/total)*100} | Corr: {correct} | Total: {total}')

    print(f'Accuracy: {(correct/total)*100} | Correct: {correct} | Total: {total}')
    return correct, total


attack_pgd = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
attack_aa = torchattacks.AutoAttack(model, eps=8/255, n_classes=10)

attack = attack_pgd




if args.case == '1':
    ## LL + HF - weights as per sensitive 
    calc_acc(testloader, model, attack, norm_ll=False, inv=False, adv=False)
    print('-'*10)
    calc_acc(testloader, model, attack, norm_ll=False, inv=False, adv=True)

elif args.case == '2':

    calc_acc(testloader, model, attack, norm_ll=False, inv=True, adv=False)
    print('-'*10)
    calc_acc(testloader, model, attack, norm_ll=False, inv=True, adv=True)

elif args.case == '3':

    calc_acc(testloader, model, attack, norm_ll=True, inv=False, adv=False)
    print('-'*10)
    calc_acc(testloader, model, attack, norm_ll=True, inv=False, adv=True)

elif args.case == '4':

    calc_acc(testloader, model, attack, norm_ll=True, inv=True, adv=False)
    print('-'*10)
    calc_acc(testloader, model, attack, norm_ll=True, inv=True, adv=True)


elif args.case == '5':

    calc_acc_full(testloader, model, attack, adv=False)
    print('-'*10)
    calc_acc_full(testloader, model, attack, adv=True)

elif args.case == '6':

    calc_acc(testloader, model, attack, norm_ll=True, inv=True, adv=False, only_ll=True)
    print('-'*10)
    calc_acc(testloader, model, attack, norm_ll=True, inv=True, adv=True, only_ll=True)



