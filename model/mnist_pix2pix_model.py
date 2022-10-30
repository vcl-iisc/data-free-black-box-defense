# DA Object pix2pix model


import torch
import torch.nn as nn
from torch.nn import init
import functools
import os.path as osp
from .base_model import BaseModel
from . import networks
from utils import get_freq, cosim_loss, get_comp_wv_batch, Res18PerceptualLoss, TV_loss, mixup_data, mixup_wavelet, spectral_loss, get_LL_batch, visualize_inputs, mixup_wavelet_custom, get_k_compfrom_HF_batch, prog_k, get_LL_batch_mnist, get_HFtop_compfrom_HF_batch_mnist,get_gn_perturbed_img_batch, get_LL_mnist_torch, load_model
import numpy as np
from util.NCEAverage import NCEAverage
from util.NCECriterion import NCECriterion
from DA.utils import load_model_office, Normalize, Trades_adv
import torch.nn.functional as F


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
            parser.add_argument('--aug', default=None, help='which augmentation to use')
            parser.add_argument('--kl_temp', type=float, default=1)
            parser.add_argument('--gn_stddev', type=float, default=0.05)


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.opt = opt
        if self.opt.reg == 'discriminator':
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.reg == 'discriminator_spectral':
            self.loss_names = ['G_Spectral', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.reg == 'onlycosim':
            self.loss_names = ['G_onlycosim', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.reg == 'cosim_KL':
            self.loss_names = ['G_onlycosim', 'G_kl', 'gen_Acc', 'og_Acc', 'lp_Acc', 'gen_adv_Acc', 'adv_Acc', 'lp_adv_Acc']
        if self.opt.reg == 'onlyperceptual':
            self.loss_names = ['G_onlyperceptual', 'gen_Acc', 'og_Acc', 'lp_Acc']  #'G_TV', 
        if self.opt.reg == 'discriminator_cosim':
            self.loss_names = ['G_onlycosim', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.reg == 'discriminator_perceptual':
            self.loss_names = [ 'G_onlyperceptual', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc'] # 'G_TV',
        if self.opt.reg == 'discriminator_teacher':
            self.loss_names = ['G_GAN', 'G_GAN_from_T', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'T_real', 'T_fake', 'T_real_Acc','T_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.instance_d==True:
            self.loss_names.append('G_instance')
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
        self.denorm = Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5]).to(self.device)
        opt.device = self.device
#         self.netF, self.netB, self.netC = load_model_office(opt)
#         self.netF.eval()
#         self.netB.eval()
#         self.netC.eval()
        self.netT = load_model(opt, load_as_D=False)
        self.netT.eval()

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterioncosim = cosim_loss
            self.criterionspectral = spectral_loss(opt)
            self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
            self.crossentropy = nn.CrossEntropyLoss()
#             criterion_kl = nn.KLDivLoss(size_average=False)



            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        self.ll_transform = get_LL_mnist_torch(lev=self.opt.decomposition_level).float().cuda()

            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        
#         if opt.progressive_rand:
#         coefficients_percent = prog_k(epoch, epoch_min=1, epoch_max=100, k_min=10, k_max=85)
        self.real_B = input[0]                 #real_B :> originalimage 
        self.real_B = self.real_B.to(self.device, dtype=torch.float)
#         print(self.real_B.size())
        self.real_adv = input[1]                 #real_B :> originalimage 
#         self.Final_model = nn.Sequential(self.ll_transform, self.norm, self.netG.module, self.denorm, self.netF,self.netB,self.netC)
#         self.Final_model.eval()
#         self.real_adv = Trades_adv(self.real_B,7,3/255,'l_inf',0.01,self.Final_model)
#         self.netF.eval()
#         self.netB.eval()
#         self.netC.eval()
#         self.netG.train()
# #         exit()
#         self.real_adv = self.real_adv.to(self.device, dtype=torch.float)
#         self.real_adv = get_gn_perturbed_img_batch(self.real_B, self.opt.gn_stddev,self.opt.decomposition_level)             #real_B :> originalimage 
        self.real_adv = self.real_adv.to(self.device, dtype=torch.float)
        
#         self.real_A = input[1].float().to(self.device)         #real_A :> LF image
        if self.opt.transform_type=='lp':
            self.real_A = get_freq(self.real_B, self.opt.lp_rad)[0].float().to(self.device)
        elif self.opt.transform_type=='wv':
            self.real_A  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
        elif self.opt.transform_type=='wv_custom': 
            self.real_A  = get_k_compfrom_HF_batch(self.real_B, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_A_adv  = get_k_compfrom_HF_batch(self.real_adv, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
#             self.real_A  = get_HFtop_compfrom_HF_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
        elif self.opt.transform_type=='ll':
            self.real_A = get_LL_batch(self.real_B, level=self.opt.decomposition_level).float().to(self.device)
            self.real_A_adv = get_LL_batch(self.real_adv, level=self.opt.decomposition_level).float().to(self.device)
        
        self.real_A = self.norm(self.real_A)     
        self.real_B = self.norm(self.real_B)
        self.real_adv = self.norm(self.real_adv)     
        self.real_A_adv = self.norm(self.real_A_adv)
        self.labels = input[2].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_adv = self.netG(self.real_A_adv)  # G(A)
#         self.fake_B = self.Final_model(self.real_B)  # G(A)
#         self.fake_B_adv = self.Final_model(self.real_adv)  # G(A)

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

    def backward_T(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netT_C(self.fake_B.detach())
        self.loss_T_fake = self.criterionGAN(pred_fake, False)
        self.T_fake_metrics['correct'] += np.sum(1 - np.round(self.sigmoid(pred_fake).detach().cpu().numpy()))
        self.T_fake_metrics['total'] += pred_fake.size(0)
        self.loss_T_fake_Acc = (self.T_fake_metrics['correct']/self.T_fake_metrics['total'])*100.
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netT_C(self.real_B)
        self.loss_T_real = self.criterionGAN(pred_real, True)
        self.T_real_metrics['correct'] += np.sum(np.round(self.sigmoid(pred_real).detach().cpu().numpy()))
        self.T_real_metrics['total'] += pred_real.size(0)
        self.loss_T_real_Acc = (self.T_real_metrics['correct']/self.T_real_metrics['total'])*100.        
        # combine loss and calculate gradients
        self.loss_T = (self.loss_T_fake + self.loss_T_real) * 0.5
        self.loss_T.backward()        
        
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.reg == 'discriminator' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
#             self.loss_G_L1_ll1 = self.criterionL1(self.fake_B_lev_1, self.real_B_lev_1) * 50
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 
        if self.opt.reg == 'discriminator_spectral' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G_Spectral = self.criterionspectral.loss(self.fake_B, self.real_B) * self.opt.lambda_reg
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Spectral
        if self.opt.reg == 'cosim_KL':
#             teacher_pred_real_B = self.netC(self.netB(self.netF(self.denorm(self.real_B))))
#             teacher_pred_fake_B = self.netC(self.netB(self.netF(self.denorm(self.fake_B))))
#             teacher_pred_fake_B_adv = self.netC(self.netB(self.netF(self.denorm(self.fake_B_adv))))
#             self.loss_G_onlycosim = self.criterioncosim(teacher_pred_fake_B ,teacher_pred_real_B.detach())
#             self.loss_G_kl = self.kl_criterion(F.log_softmax(teacher_pred_fake_B_adv/self.opt.kl_temp, dim=1),
#                              F.softmax(teacher_pred_fake_B.detach()/self.opt.kl_temp, dim=1)) * (self.opt.kl_temp * self.opt.kl_temp)
#             _, pred = torch.max(teacher_pred_real_B, 1)
#             self.labels = pred
#             self.loss_G_CE = self.crossentropy(teacher_pred_fake_B_adv, pred)
#             self.loss_G = self.loss_G_onlycosim + self.loss_G_kl + self.loss_G_CE
            teacher_pred_real_B = self.netT(self.denorm(self.real_B))
            teacher_pred_fake_B = self.netT(self.denorm(self.fake_B))
            teacher_pred_fake_B_adv = self.netT(self.denorm(self.fake_B_adv))
            self.loss_G_onlycosim = self.criterioncosim(teacher_pred_fake_B ,teacher_pred_real_B.detach())
            self.loss_G_kl = self.kl_criterion(F.log_softmax(teacher_pred_fake_B_adv/self.opt.kl_temp, dim=1),
                             F.softmax(teacher_pred_fake_B.detach()/self.opt.kl_temp, dim=1)) * (self.opt.kl_temp * self.opt.kl_temp)
            _, pred = torch.max(teacher_pred_real_B, 1)
            self.labels = pred
            self.loss_G_CE = self.crossentropy(teacher_pred_fake_B_adv, pred)
            self.loss_G = self.loss_G_onlycosim + self.loss_G_kl + self.loss_G_CE
        if self.opt.reg == 'discriminator_teacher' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            pred_fake_T = self.netT_C(self.fake_B)
            self.loss_G_GAN_from_T = self.criterionGAN(pred_fake_T, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_GAN_from_T           
        if self.opt.reg == 'onlycosim':
            teacher_pred_real_B = self.netC(self.netB(self.netF(self.denorm(self.real_B))))
            teacher_pred_fake_B = self.netC(self.netB(self.netF(self.denorm(self.fake_B))))
            self.loss_G_onlycosim = self.criterioncosim(teacher_pred_fake_B ,teacher_pred_real_B.detach())
            self.loss_G = self.loss_G_onlycosim
        if self.opt.reg == 'onlyperceptual':
            self.loss_G_onlyperceptual = self.criterionperceptual(self.denorm(self.fake_B), self.denorm(self.real_B).detach())
#             self.loss_G_TV = TV_loss(self.denorm(self.fake_B))*0.0001  + self.loss_G_TV
            self.loss_G = self.loss_G_onlyperceptual
        if self.opt.reg == 'discriminator_cosim' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # combine loss and calculate gradients
            teacher_pred_real_B = self.netT(self.denorm(self.real_B))
            teacher_pred_fake_B = self.netT(self.denorm(self.fake_B))
            self.loss_G_onlycosim = self.criterioncosim(teacher_pred_fake_B ,teacher_pred_real_B.detach()) * self.opt.lambda_reg 
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_onlycosim
        if self.opt.reg == 'discriminator_perceptual' :
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G_onlyperceptual = self.criterionperceptual(self.denorm(self.fake_B), self.denorm(self.real_B).detach())* self.opt.lambda_reg 
#             self.loss_G_TV = TV_loss(self.denorm(self.fake_B))*0.0005  + self.loss_G_TV
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_onlyperceptual
        
#         if self.opt.instance_d==True:
#             outputs = self.lemniscate(self.instance_feat, self.indexes)
#             loss_G_instance = self.criterion(outputs, self.indexes)
#             self.loss_G  += loss_G_instance


        self.loss_G.backward()
#         exit()

    def optimize_parameters(self, opt_G=True, opt_D=True):
        self.forward()                   # compute fake images: G(A)
        # update D
        if opt_D :
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
#             if self.opt.reg == 'discriminator_teacher' :
#                 self.set_requires_grad(self.netT_C[1].linear, True)  # enable backprop for D
#                 self.optimizer_T_C.zero_grad()     # set D's gradients to zero
#                 self.backward_T()                # calculate gradients for D
#                 self.optimizer_T_C.step()          # update D's weights
        
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
        self.lp_adv_metrics = {'correct':0, 'total':0}
        self.gen_adv_metrics = {'correct':0, 'total':0}
        self.adv_metrics = {'correct':0, 'total':0}
        
    def reinit_D_metrics(self):
        self.D_fake_metrics = {'correct':0, 'total':0}
        self.D_real_metrics = {'correct':0, 'total':0} 
        if self.opt.reg == 'discriminator_teacher' :
            self.T_fake_metrics = {'correct':0, 'total':0}
            self.T_real_metrics = {'correct':0, 'total':0} 
        
    def set_toy_imgs(self, dataloader):
        for data, label in dataloader:
            break
        self.toy_real_A = data[0].unsqueeze(0).to(self.device)
        if self.opt.transform_type=='lp':
            self.toy_real_B = get_freq(self.toy_real_A, self.opt.lp_rad)[0].float().to(self.device)
        elif self.opt.transform_type=='wv':
            self.toy_real_B  = get_comp_wv_batch(self.toy_real_A).float().to(self.device)
        elif self.opt.transform_type=='ll':
            self.toy_real_B  = get_LL_batch(self.toy_real_A, level=self.opt.decomposition_level).float().to(self.device)
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
#             self.netF.eval()
#             self.netB.eval()
#             self.netC.eval()
            self.netT.eval()
            
            # Accuracy on Reconstructed  
            self.teach_fake_B_pred = self.netT(self.denorm(self.fake_B))
            _, pred = torch.max(self.teach_fake_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_fake_B_pred.size(0)
            self.loss_gen_Acc = (self.gen_metrics['correct']/self.gen_metrics['total']) * 100.


            # Accuracy on Low Pass
            self.teach_real_A_pred = self.netT(self.denorm(self.real_A))
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_pred.size(0)
            self.loss_lp_Acc = (self.lp_metrics['correct']/self.lp_metrics['total']) * 100.


            ## Accuracy on clean
            self.teach_real_B_pred = self.netT(self.denorm(self.real_B))
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)  
            self.loss_og_Acc = (self.og_metrics['correct']/self.og_metrics['total']) * 100.
            
            
            self.teach_fake_B_adv_pred = self.netT(self.denorm(self.fake_B_adv))
            _, pred = torch.max(self.teach_fake_B_adv_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_adv_metrics['correct'] += correct
            self.gen_adv_metrics['total']   += self.teach_fake_B_adv_pred.size(0)
            self.loss_gen_adv_Acc = (self.gen_adv_metrics['correct']/self.gen_adv_metrics['total']) * 100.


            # Accuracy on Low Pass
            self.teach_real_A_adv_pred = self.netT(self.denorm(self.real_A_adv))
            _, pred = torch.max(self.teach_real_A_adv_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_adv_metrics['correct'] += correct
            self.lp_adv_metrics['total']   += self.teach_real_A_adv_pred.size(0)
            self.loss_lp_adv_Acc = (self.lp_adv_metrics['correct']/self.lp_adv_metrics['total']) * 100.


            ## Accuracy on clean
            self.teach_real_adv_pred = self.netT(self.denorm(self.real_adv))
            _, pred = torch.max(self.teach_real_adv_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_adv_pred.size(0)  
            self.loss_adv_Acc = (self.adv_metrics['correct']/self.adv_metrics['total']) * 100.