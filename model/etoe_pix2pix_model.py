## IDWT(LL Level 2) -> I2I -> LL Level 1 

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
import functools
from .base_model import BaseModel
from . import networks
from utils import load_model, Normalize, spectral_loss, get_ll_orig_batch, get_HFmasked_ll_batch, get_ll_orig_inverse_batch, get_ll_orig_inverse_batch8, get_HFtop_comp_batch, get_HFtop_comp_lev_batch, get_HFtop_comp_lev_ind_batch, get_HFtop_comp_inv_batch , get_HFtop_comp_inv_torch_batch, get_ll_orig_inverse_torch_batch
import numpy as np
from util.NCEAverage import NCEAverage
from util.NCECriterion import NCECriterion

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_reg', type=float, default=5.0, help='weight for reg loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.opt = opt
        if self.opt.reg == 'Spectral':
            self.loss_names = ['G_L1', 'gen_Acc', 'og_Acc', 'lp_Acc'] #
        if self.opt.reg == 'discriminator':
            self.loss_names = ['G1_GAN', 'G1_L1','G2_GAN', 'G2_L1', 'D1_real', 'D1_fake', 'D1_real_Acc','D1_fake_Acc', 'gen1_Acc', 'og1_Acc', 'lp1_Acc', 'D2_real', 'D2_fake', 'D2_real_Acc','D2_fake_Acc', 'gen2_Acc', 'og2_Acc', 'lp2_Acc']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['toy_real_A', 'toy_fake_B', 'toy_real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1', 'D2' ]
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_16', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,instance_d=self.opt.instance_d)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,instance_d=self.opt.instance_d)
        self.sigmoid = torch.nn.Sigmoid()
        self.opt = opt
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        self.denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),])
        opt.device = self.device
        self.netT = load_model(opt, load_as_D=False)
        self.netT.eval()

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic_16',
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
#             self.criterioncosim = cosim_loss
            self.criterionspectral = spectral_loss(opt)
#             self.criterionperceptual = Res18PerceptualLoss(self.netT[1],opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=0.00005, betas=(opt.beta1, 0.999), weight_decay=0.0001)
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=0.00005, betas=(opt.beta1, 0.999), weight_decay=0.0001)
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D2)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_B2 = input[0].to(self.device)                 #real_B :> original image
#         print(self.real_B2.size())
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_B1 = get_ll_orig_batch(self.real_B2,lev=1).float().to(self.device)/2 
#         print(self.real_B1.size())
#         print(torch.min(self.real_B), torch.max(self.real_B))
#         self.real_A1 = get_HFtop_comp_batch(self.real_B2, lev=1, keep = 15/100).float().to(self.device)/2  # ll' l1
        self.real_A1 = get_HFmasked_ll_batch(self.real_B2, lev=1).float().to(self.device)/2  # ll' l1

#         print(self.real_A1.size())
#         exit()
#         print(torch.min(self.real_A), torch.max(self.real_A))
#         self.real_B2 = input[0].to(self.device)                 #real_B :> original image
        self.real_B2 = self.norm(self.real_B2)
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_A1 = self.norm(self.real_A1)     
#         print(torch.min(self.real_A), torch.max(self.real_A))
        self.real_B1 = self.norm(self.real_B1)
#         print(torch.min(self.real_B), torch.max(self.real_B))

        self.labels = input[1].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B1 = self.netG1(self.real_A1)  # G(A)
#         self.real_A2 = get_HFtop_comp_inv_torch_batch(self.opt,self.real_B2,self.denorm(self.fake_B1)*2,keep=0.15,lev=1).float().to(self.device) 
        self.real_A2 = get_ll_orig_inverse_torch_batch(self.opt,self.denorm(self.fake_B1)*2).float().to(self.device) 
        self.fake_B2 = self.netG2(self.real_A2)  # G(A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB1 = torch.cat((self.real_A1, self.fake_B1), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD1(fake_AB1.detach())
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)
        self.D1_fake_metrics['correct'] += np.sum(np.round(np.average(np.average(1 - np.round(self.sigmoid(pred_fake).detach().cpu().numpy()),3),2)))
        self.D1_fake_metrics['total'] += pred_fake.size(0)
        self.loss_D1_fake_Acc = (self.D1_fake_metrics['correct']/self.D1_fake_metrics['total'])*100.
        # Real
        real_AB1 = torch.cat((self.real_A1, self.real_B1), 1)
        pred_real = self.netD1(real_AB1)
        self.loss_D1_real = self.criterionGAN(pred_real, True)
        self.D1_real_metrics['correct'] += np.sum(np.round(np.average(np.average(np.round(self.sigmoid(pred_real).detach().cpu().numpy()),3),2)))
        self.D1_real_metrics['total'] += pred_real.size(0)
        self.loss_D1_real_Acc = (self.D1_real_metrics['correct']/self.D1_real_metrics['total'])*100. 
        
        fake_AB2 = torch.cat((self.real_A2, self.fake_B2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD2(fake_AB2.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)
        self.D2_fake_metrics['correct'] += np.sum(np.round(np.average(np.average(1 - np.round(self.sigmoid(pred_fake).detach().cpu().numpy()),3),2)))
        self.D2_fake_metrics['total'] += pred_fake.size(0)
        self.loss_D2_fake_Acc = (self.D2_fake_metrics['correct']/self.D2_fake_metrics['total'])*100.
        # Real
        real_AB2 = torch.cat((self.real_A2, self.real_B2), 1)
        pred_real = self.netD2(real_AB2.detach())
        self.loss_D2_real = self.criterionGAN(pred_real, True)
        self.D2_real_metrics['correct'] += np.sum(np.round(np.average(np.average(np.round(self.sigmoid(pred_real).detach().cpu().numpy()),3),2)))
        self.D2_real_metrics['total'] += pred_real.size(0)
        self.loss_D2_real_Acc = (self.D2_real_metrics['correct']/self.D2_real_metrics['total'])*100. 
        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5        
        self.loss_D2.backward()
        self.loss_D1.backward()
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.reg == 'discriminator' :
            fake_AB1 = torch.cat((self.real_A1, self.fake_B1), 1)
            fake_AB2 = torch.cat((self.real_A2, self.fake_B2), 1)
            pred_fake1 = self.netD1(fake_AB1)            
            pred_fake2 = self.netD2(fake_AB2)
            self.loss_G2_GAN = self.criterionGAN(pred_fake2, True)
            self.loss_G1_GAN = self.criterionGAN(pred_fake1, True) 
            # Second, G(A) = 
            self.loss_G2_L1=(self.criterionL1(self.fake_B2, self.real_B2)) * self.opt.lambda_L1
            self.loss_G1_L1=(self.criterionL1(self.fake_B1, self.real_B1)) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G1 = self.loss_G1_GAN + self.loss_G1_L1
            self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L1
        if self.opt.reg == 'Spectral' :
#             self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            #self.loss_G_Spectral = self.criterionspectral.loss(self.fake_B, self.real_B)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1
        self.loss_G2.backward(retain_graph=True)
        self.loss_G1.backward()        

    def optimize_parameters(self, opt_G=True, opt_D=True):
        self.forward()                   # compute fake images: G(A)
        # update D
        if opt_D :
            self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
            self.optimizer_D1.zero_grad()     # set D's gradients to zero
            self.optimizer_D2.zero_grad()     # set D's gradients to zero            
            self.backward_D()                # calculate gradients for D
            self.optimizer_D1.step()          # update D's weights
            self.optimizer_D2.step()          # update D's weights
            
        # update G
        if opt_G:
            self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
            self.optimizer_G1.zero_grad()        # set G's gradients to zero
            self.optimizer_G2.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G1.step()             # udpate G's weights
            self.optimizer_G2.step()             # udpate G's weights

    def init_metrics(self):
        self.lp1_metrics = {'correct':0, 'total':0}
        self.gen1_metrics = {'correct':0, 'total':0}
        self.og1_metrics = {'correct':0, 'total':0}  
        self.lp2_metrics = {'correct':0, 'total':0}
        self.gen2_metrics = {'correct':0, 'total':0}
        self.og2_metrics = {'correct':0, 'total':0}  
        
        
    def reinit_D_metrics(self):
        self.D1_fake_metrics = {'correct':0, 'total':0}
        self.D1_real_metrics = {'correct':0, 'total':0} 
        self.D2_fake_metrics = {'correct':0, 'total':0}
        self.D2_real_metrics = {'correct':0, 'total':0} 
        
    def set_toy_imgs(self, dataloader):
        for data, label in dataloader:
            break
        self.toy_real_B = get_HFmasked_ll_batch(data,lev=1)[0].unsqueeze(0).float().to(self.device)/2
        self.toy_real_A  = get_ll_orig_batch(data,lev=1)[0].unsqueeze(0).float().to(self.device)/2
        self.toy_real_A = self.norm(self.toy_real_A)
        self.toy_real_B = self.norm(self.toy_real_B)
        self.toy_fake_B = self.netG(self.toy_real_B)   

    
#     def denorm(self, x):
#         batch_size = x.size(0)
#         x_new = torch.zeros(x.size(), dtype=x.dtype)
#         for idx in range(batch_size):
#             x_new[idx] = ((x[idx] - torch.min(x[idx]))/(torch.max(x[idx])-torch.min(x[idx])))
#         x_new = x_new.to(self.device)
#         return x_new
        
    def eval_teacher(self):
               
        with torch.no_grad():
            # Accuracy on Reconstructed  
            self.teach_fake_B1_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.fake_B1)*2).float().to(self.device))
            _, pred = torch.max(self.teach_fake_B1_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen1_metrics['correct'] += correct
            self.gen1_metrics['total']   += self.teach_fake_B1_pred.size(0)
            self.loss_gen1_Acc = (self.gen1_metrics['correct']/self.gen1_metrics['total']) * 100.


            # Accuracy on Low Pass
            self.teach_real_A1_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.real_A1)*2).float().to(self.device))
            _, pred = torch.max(self.teach_real_A1_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp1_metrics['correct'] += correct
            self.lp1_metrics['total']   += self.teach_real_A1_pred.size(0)
            self.loss_lp1_Acc = (self.lp1_metrics['correct']/self.lp1_metrics['total']) * 100.


            ## Accuracy on clean
            self.teach_real_B1_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.real_B1)*2).float().to(self.device))
            _, pred = torch.max(self.teach_real_B1_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og1_metrics['correct'] += correct
            self.og1_metrics['total']   += self.teach_real_B1_pred.size(0)  
            self.loss_og1_Acc = (self.og1_metrics['correct']/self.og1_metrics['total']) * 100. 
            
            # Accuracy on Reconstructed  
            self.teach_fake_B2_pred = self.netT(self.denorm(self.fake_B2))
            _, pred = torch.max(self.teach_fake_B2_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen2_metrics['correct'] += correct
            self.gen2_metrics['total']   += self.teach_fake_B2_pred.size(0)
            self.loss_gen2_Acc = (self.gen2_metrics['correct']/self.gen2_metrics['total']) * 100.


            # Accuracy on Low Pass
            self.teach_real_A2_pred = self.netT(self.denorm(self.real_A2))
            _, pred = torch.max(self.teach_real_A2_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp2_metrics['correct'] += correct
            self.lp2_metrics['total']   += self.teach_real_A2_pred.size(0)
            self.loss_lp2_Acc = (self.lp2_metrics['correct']/self.lp2_metrics['total']) * 100.


            ## Accuracy on clean
            self.teach_real_B2_pred = self.netT(self.denorm(self.real_B2))
            _, pred = torch.max(self.teach_real_B2_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og2_metrics['correct'] += correct
            self.og2_metrics['total']   += self.teach_real_B2_pred.size(0)  
            self.loss_og2_Acc = (self.og2_metrics['correct']/self.og2_metrics['total']) * 100. 
                
                