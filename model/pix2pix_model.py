# orig pix2pix model

import torch
import torch.nn as nn
from torch.nn import init
import functools
from .base_model import BaseModel
from . import networks
from utils import Denormalize, load_model, get_freq, Normalize, cosim_loss, get_comp_wv_batch, Res18PerceptualLoss, TV_loss, load_model_1, mixup_data, mixup_wavelet, spectral_loss, get_LL_batch, visualize_inputs, mixup_wavelet_custom, get_k_compfrom_HF_batch, prog_k, wvlt_transform
import numpy as np
from util.NCEAverage import NCEAverage
from util.NCECriterion import NCECriterion
import torch.nn.functional as F
import torchattacks

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
            self.loss_names = ['G_onlycosim', 'G_kl','gen_Acc', 'og_Acc', 'lp_Acc' ]

        if self.opt.reg == 'kl':
            self.loss_names = ['G_kl','gen_Acc', 'og_Acc', 'lp_Acc' ]

        if self.opt.reg == 'onlyperceptual':
            self.loss_names = ['G_onlyperceptual', 'gen_Acc', 'og_Acc', 'lp_Acc']  #'G_TV', 
        if self.opt.reg == 'discriminator_cosim':
            self.loss_names = ['G_onlycosim', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        if self.opt.reg == 'discriminator_perceptual':
            self.loss_names = [ 'G_onlyperceptual', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc'] # 'G_TV',
        if self.opt.reg == 'discriminator_teacher':
            self.loss_names = ['G_GAN', 'G_GAN_from_T', 'G_L1', 'D_real', 'D_fake', 'D_real_Acc','D_fake_Acc', 'T_real', 'T_fake', 'T_real_Acc','T_fake_Acc', 'gen_Acc', 'og_Acc', 'lp_Acc']
        
        #this is used when only wc loss is used 
        if self.opt.reg == "":
            self.loss_names = []

        if self.opt.wc_loss is not None:   # use wavelet consitency loss
            self.loss_names.append("G_wc")
            
        

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
        #self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        #self.denorm = Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5]).to(self.device)
        #self.norm =  Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]).to(self.device)
        
        self.norm= Normalize(mean=[0.5], std=[0.5]).to(self.device)
        #self.denorm= Normalize(mean=[-0.5/0.5], std=[1/0.5]).to(self.device)
        self.denorm = Denormalize(mean=[0.5], std=[0.5]).to(self.device)
        
        opt.device = self.device
        if opt.surrogate_training:
            self.netT = load_model_1(opt.surrogate_teacher_model,opt.surrogate_load_path,opt.device, load_as_D=False, input_channels=opt.input_nc)
        else:
            self.netT = load_model(opt, load_as_D=False)
        self.netT.eval()

        #victim model is against which testing is done.
        #since this will be a black box model, we use self.netT( surrogate model) to generate the adversarial n_samples

        
#         if self.opt.reg == 'discriminator_teacher' :
#             self.netT_C = load_model(opt, load_as_D=True)
#             self.netT_C.eval()
#             self.netT_C[1].linear.train()
#             self.optimizer_T_C = torch.optim.Adam(self.netT_C[1].linear.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_T_C)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.down =nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=True), nn.InstanceNorm2d(64)).to(self.device)
#             self.up = nn.Sequential(nn.ReLU(True), nn.Conv2d(128, 3, kernel_size=4, padding=1, stride=1, bias=True), nn.Tanh()).to(self.device)
#             print(list(self.netG.parameters())+list(self.down.parameters())+list(self.up.parameters()))
#             print(self.down.parameters())
#             print(self.up.parameters())
#             exit()
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterioncosim = cosim_loss
            self.criterionspectral = spectral_loss(opt)
            self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
            self.crossentropy = nn.CrossEntropyLoss()

#             self.criterionperceptual = Res18PerceptualLoss(self.netT[1],opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            ## Define Instance-Discrimination Losses
#             nce_k = 4096
#             nce_t = 0.07
#             nce_m = 0.5
#             ndata = opt.len_data

#             #self.lemniscate = NCEAverage(low_dim=128, ndata=ndata, nce_k=nce_k, nce_t=nce_t, nce_m=nce_m)
#             self.lemniscate = NCEAverage(inputSize = 128, outputSize = ndata, K = nce_k, T=nce_t, momentum=nce_m).to(self.device)
#             self.instane_criterion = NCECriterion(ndata).to(self.device)
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
#             conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

#             self.net_G2 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=True), nn.InstanceNorm2d(64), self.netG.module.model.model[1], nn.ReLU(True), nn.Conv2d(128, 3, kernel_size=4, padding=1, stride=1, bias=True), nn.Tanh()).to(self.device)
            
#             print(self.net_G2)
#             exit()


                    
            
               #radius mask for fourier based image filtering
        self.mask = None
                

    def set_input(self, input, epoch=0):  # epoch is not used, but on removing this argument older client code will break , so keeping it with default value
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        
#         if opt.progressive_rand:
#         coefficients_percent = prog_k(epoch, epoch_min=1, epoch_max=100, k_min=10, k_max=85)
        self.real_B = input[0]                 #real_B :> originalimage 
        # idx=4
        # visualize_inputs(self.real_B[idx], path='mixup_wv_og_before_mixup_4.png')

#         if self.opt.aug =='mixup':
#             self.real_B = mixup_data(self.real_B,alpha=0.25,use_cuda=False)
#             # print(f'using mixup, {self.real_B.shape}')
        
#         if self.opt.aug=='mixup_wavelet':
#             self.real_B = mixup_wavelet(self.real_B, lev=self.opt.decomposition_level, use_cuda=False)
#             # print(f'using mixup wv, {self.real_B.shape}')
            
#         if self.opt.aug=='mixup_wavelet_custom':
#             self.real_B = mixup_wavelet_custom(self.real_B, lev=self.opt.decomposition_level, use_cuda=False)

#         self.real_B_gt = torch.cat((self.real_B, self.real_B), 0)

        # print(f'Batch Size: {self.real_B.shape} | idx: {idx}')
        # visualize_inputs(self.real_B[idx], path='mixup_wv_og_4.png')
        # visualize_inputs(self.real_B[idx + self.opt.batch_size], path='mixup_wv_aug_4.png')
        # exit(0)
        
        #load clear and adversarial image
        self.labels = input[-1].to(self.device)
       
        #move to cuda device
        self.real_B = self.real_B.to(self.device, dtype=torch.float)  
       
        #this is temporary comment to check performance of I2I without wavelet
        #frequency based transformation
        if self.opt.transform_type=='lp':
            if self.mask is None:
                rows, cols = input[0].size(2) , input[0].size(3)
                mask = torch.zeros((rows, cols)).to(self.device)
                for i in range(rows):
                    for j in range(cols):
                        dis = np.sqrt((i - rows/2) ** 2 + (j - rows/2) ** 2)
                        if dis < self.opt.lp_rad:
                            mask[i, j] = 1.0
                
                self.mask = mask
                
            self.real_A = get_freq(self.real_B, self.mask)[0].float()
            
        elif self.opt.transform_type=='wv':
            # print(f'using wavelet transform: {self.opt.decomposition_level}, {self.opt.coefficients_percent}')
            self.real_A  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)   
        
        elif self.opt.transform_type in ["wv_custom" , "visushrink" , 'bayes','rigsure']:
            #self.real_A  = get_k_compfrom_HF_batch(self.real_B, setting=self.opt.which_k, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_A = wvlt_transform(self.real_B, device  = self.device,lev=self.opt.decomposition_level,setting=self.opt.which_k,
                                   keep = self.opt.coefficients_percent/100,selection_strategy=self.opt.transform_type , wvlt =self.opt.wvlt, threshold_strategy=self.opt.threshold_strategy).float().to(self.device)
        elif self.opt.transform_type=='ll':
            #self.real_A = get_LL_batch(self.real_B, level=self.opt.decomposition_level).float().to(self.device)
            self.real_A = wvlt_transform(self.real_B, device  = self.device,lev=self.opt.decomposition_level,selection_strategy="ll" , wvlt = self.opt.wvlt).float().to(self.device)
#         self.real_A_lev_1 = self.norm(self.real_A_lev_1)  
        #self.real_A = self.real_B   #remove this line later once you uncomment above code
        self.real_A = self.norm(self.real_A)     
        self.real_B = self.norm(self.real_B)

        self.real_adv_B = None
        if len(input)> 2:
            self.real_adv_B = input[1]
            
            #do similar operation of adversarial image as that of clean image if it is not null
            if self.real_adv_B is not None:
                self.real_adv_B = self.real_adv_B.to(self.device,dtype=torch.float)
                #frequency based transformation
                if self.opt.transform_type=='lp':
                    self.real_adv_A = get_freq(self.real_adv_B, self.mask)[0].float()
                    #self.real_adv_A = get_freq(self.real_adv_B, self.opt.lp_rad)[0].float().to(self.device)
                elif self.opt.transform_type=='wv':
                    # print(f'using wavelet transform: {self.opt.decomposition_level}, {self.opt.coefficients_percent}')
                    self.real_adv_A  = get_comp_wv_batch(self.real_adv_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)   
                elif self.opt.transform_type in ["wv_custom" , "visushrink", 'bayes','rigsure']:
                    #self.real_adv_A  = get_k_compfrom_HF_batch(self.real_adv_B, setting=self.opt.which_k, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
                    self.real_adv_A = wvlt_transform(self.real_adv_B, device  = self.device,lev=self.opt.decomposition_level,setting=self.opt.which_k,
                                   keep = self.opt.coefficients_percent/100,selection_strategy=self.opt.transform_type ,wvlt = self.opt.wvlt , threshold_strategy=self.opt.threshold_strategy).float().to(self.device)
                elif self.opt.transform_type=='ll':
                    #self.real_adv_A = get_LL_batch(self.real_adv_B, level=self.opt.decomposition_level).float().to(self.device)
                    self.real_adv_A = wvlt_transform(self.real_adv_B, device  = self.device,lev=self.opt.decomposition_level,selection_strategy="ll",wvlt = self.opt.wvlt).float().to(self.device)
                
                #self.real_adv_A= self.real_adv_B  # comment this line once you uncomment above block
                self.real_adv_A= self.norm(self.real_adv_A)
                self.real_adv_B = self.norm(self.real_adv_B)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.fake_B = self.netG(self.real_A)  # G(A)
        
        if self.real_adv_B is not None:
            self.fake_adv_B = self.netG(self.real_adv_A)

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
            teacher_pred_real_B = self.netT(self.denorm(self.real_B))
            teacher_pred_fake_B = self.netT(self.denorm(self.fake_B))
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
        
        if self.opt.reg == 'cosim_KL' or self.opt.reg=="kl":
            #Completion CS( M (T (LL_c)),M (C)) + KL ( M(T (LL_c)), M(T (LL_adv))  # formula
            teacher_pred_real_B = self.netT(self.denorm(self.real_B))
            teacher_pred_fake_B = self.netT(self.denorm(self.fake_B))

            teacher_pred_fake_adv_B = self.netT(self.denorm(self.fake_adv_B))
            
            self.loss_G_kl = self.kl_criterion(F.log_softmax(teacher_pred_fake_adv_B/self.opt.kl_temp, dim=1),
                                F.softmax(teacher_pred_fake_B.detach()/self.opt.kl_temp, dim=1)) * (self.opt.kl_temp * self.opt.kl_temp)
                    
            if self.opt.reg == 'cosim_KL':
                self.loss_G_onlycosim = self.criterioncosim(teacher_pred_fake_B ,teacher_pred_real_B.detach())
                self.loss_G =  self.loss_G_kl + self.loss_G_onlycosim
                                
            
            
            #_, pred = torch.max(teacher_pred_real_B, 1)
            #self.labels = pred
            #self.loss_G_CE = self.crossentropy(teacher_pred_fake_adv_B, pred)
            
            if self.opt.reg == "kl":
                self.loss_G = self.loss_G_kl


        

        if self.opt.wc_loss is not None:  
          
            #if wc parameter is defined, ad wavelet consistency loss in final loss
            if self.opt.wc_loss == "l1":
                self.loss_G_wc = self.criterionL1(self.fake_B, self.real_B.detach())
            else:
                error_msg = "incorrect wavelet consistency loss {}".format(self.opt.wc_loss)
                raise Exception(error_msg)
            if self.opt.reg == "":  #only wc loss is used
                self.loss_G = self.loss_G_wc
            else:
                self.loss_G += self.loss_G_wc


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
        
    def reinit_D_metrics(self):
        self.D_fake_metrics = {'correct':0, 'total':0}
        self.D_real_metrics = {'correct':0, 'total':0} 
        if self.opt.reg == 'discriminator_teacher' :
            self.T_fake_metrics = {'correct':0, 'total':0}
            self.T_real_metrics = {'correct':0, 'total':0} 
        
    def set_toy_imgs(self, dataloader):
        for data in dataloader:
            break

        data = data[0]
        
        self.toy_real_B = data[0].unsqueeze(0).to(self.device)
        
        if self.opt.transform_type=='lp':
            #self.toy_real_A = get_freq(self.toy_real_B, self.opt.lp_rad)[0].float().to(self.device)
            if self.mask is None:
                rows, cols = data[0].size(1) , data[0].size(2)
                mask = torch.zeros((rows, cols)).to(self.device)
                for i in range(rows):
                    for j in range(cols):
                        dis = np.sqrt((i - rows/2) ** 2 + (j - rows/2) ** 2)
                        if dis < self.opt.lp_rad:
                            mask[i, j] = 1.0
                
                self.mask = mask
            
            self.toy_real_A = get_freq(self.toy_real_B, self.mask)[0].float()
            
            #self.toy_real_A = get_freq(self.toy_real_B, self.opt.lp_rad)[0].float().to(self.device)

        elif self.opt.transform_type=='wv':
            self.toy_real_A  = get_comp_wv_batch(self.toy_real_B).float().to(self.device)
        elif self.opt.transform_type in ["wv_custom" , "visushrink", 'bayes','rigsure']:
            #self.toy_real_A  = get_comp_wv_batch(self.toy_real_B).float().to(self.device)
            self.toy_real_A = wvlt_transform(self.toy_real_B, device  = self.device,lev=self.opt.decomposition_level,setting=self.opt.which_k,
                                   keep = self.opt.coefficients_percent/100,selection_strategy=self.opt.transform_type,wvlt = self.opt.wvlt,threshold_strategy=self.opt.threshold_strategy).float().to(self.device)
        elif self.opt.transform_type=='ll':
            self.toy_real_A  = get_LL_batch(self.toy_real_B, level=self.opt.decomposition_level).float().to(self.device)
        

        self.toy_real_A = self.norm(self.toy_real_A)
        self.toy_real_B = self.norm(self.toy_real_B)
        self.toy_fake_B = self.netG(self.toy_real_A)    
    
    #Following MNIST_pix2pix model code, denorm function is replace with self.denorm layer defined in constructor
    """def denorm(self, x):
        batch_size = x.size(0)
        x_new = torch.zeros(x.size(), dtype=x.dtype)
        for idx in range(batch_size):
            x_new[idx] = ((x[idx] - torch.min(x[idx]))/(torch.max(x[idx])-torch.min(x[idx])))
        x_new = x_new.to(self.device)
        return x_new"""
        

        
    def eval_teacher(self):
               
        with torch.no_grad():
            # Accuracy on Reconstructed  
            self.teach_fake_B_pred = self.netT(self.denorm(self.fake_B))[:self.opt.batch_size]
            _, pred = torch.max(self.teach_fake_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_fake_B_pred.size(0)
            self.loss_gen_Acc = (self.gen_metrics['correct']/self.gen_metrics['total']) * 100.

            # Accuracy on Low Pass
            self.teach_real_A_pred = self.netT(self.denorm(self.real_A))[:self.opt.batch_size]
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_pred.size(0)
            self.loss_lp_Acc = (self.lp_metrics['correct']/self.lp_metrics['total']) * 100.

            ## Accuracy on clean
            self.teach_real_B_pred = self.netT(self.denorm(self.real_B))[:self.opt.batch_size]
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)  
            self.loss_og_Acc = (self.og_metrics['correct']/self.og_metrics['total']) * 100.
            
