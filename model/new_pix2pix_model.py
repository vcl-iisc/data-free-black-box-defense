## IDWT(LL Level 2) -> I2I -> LL Level 1 

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
import functools
from .base_model import BaseModel
from . import networks
from utils import load_model, Normalize, spectral_loss, get_ll_orig_batch, get_HFmasked_ll_batch, get_ll_orig_inverse_batch, get_ll_orig_inverse_batch8, get_HFtop_comp_batch, get_HFtop_comp_lev_batch, get_HFtop_comp_lev_ind_batch, get_HFtop_compfrom_HF_batch 
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
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['toy_real_A', 'toy_fake_B', 'toy_real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
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
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
#             self.criterioncosim = cosim_loss
            self.criterionspectral = spectral_loss(opt)
#             self.criterionperceptual = Res18PerceptualLoss(self.netT[1],opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_B_orig = input[0].to(self.device)                 #real_B :> original image
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_B = get_ll_orig_batch(self.real_B_orig,lev=1).float().to(self.device)/2 
#         print(torch.min(self.real_B), torch.max(self.real_B))
        self.real_A = get_HFtop_compfrom_HF_batch(self.real_B_orig, lev=1, keep = 0.1).float().to(self.device)/2  # ll' l1
#         print(torch.min(self.real_A), torch.max(self.real_A))
        self.real_B_orig = self.norm(self.real_B_orig)
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_A = self.norm(self.real_A)     
#         print(torch.min(self.real_A), torch.max(self.real_A))
        self.real_B = self.norm(self.real_B)
#         print(torch.min(self.real_B), torch.max(self.real_B))

        self.labels = input[1].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.D_fake_metrics['correct'] += np.sum(np.round(np.average(np.average(1 - np.round(self.sigmoid(pred_fake).detach().cpu().numpy()),3),2)))
        self.D_fake_metrics['total'] += pred_fake.size(0)
        self.loss_D_fake_Acc = (self.D_fake_metrics['correct']/self.D_fake_metrics['total'])*100.
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.D_real_metrics['correct'] += np.sum(np.round(np.average(np.average(np.round(self.sigmoid(pred_real).detach().cpu().numpy()),3),2)))
        self.D_real_metrics['total'] += pred_real.size(0)
        self.loss_D_real_Acc = (self.D_real_metrics['correct']/self.D_real_metrics['total'])*100.        
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.reg == 'discriminator' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.opt.reg == 'Spectral' :
#             self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            #self.loss_G_Spectral = self.criterionspectral.loss(self.fake_B, self.real_B)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, opt_G=True, opt_D=True):
        self.forward()                   # compute fake images: G(A)
        # update D
        if opt_D :
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        
        # update G
        if opt_G:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights

    def init_metrics(self):
        self.lp_metrics = {'correct':0, 'total':0}
        self.gen_metrics = {'correct':0, 'total':0}
        self.og_metrics = {'correct':0, 'total':0}   
        
    def reinit_D_metrics(self):
        self.D_fake_metrics = {'correct':0, 'total':0}
        self.D_real_metrics = {'correct':0, 'total':0} 

        
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
            self.teach_fake_B_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.fake_B)*2).float().to(self.device))
            _, pred = torch.max(self.teach_fake_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_fake_B_pred.size(0)
            self.loss_gen_Acc = (self.gen_metrics['correct']/self.gen_metrics['total']) * 100.


            # Accuracy on Low Pass
            self.teach_real_A_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.real_A)*2).float().to(self.device))
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_pred.size(0)
            self.loss_lp_Acc = (self.lp_metrics['correct']/self.lp_metrics['total']) * 100.


            ## Accuracy on clean
            self.teach_real_B_pred = self.netT(get_ll_orig_inverse_batch(self.denorm(self.real_B)*2).float().to(self.device))
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)  
            self.loss_og_Acc = (self.og_metrics['correct']/self.og_metrics['total']) * 100. 
                
                