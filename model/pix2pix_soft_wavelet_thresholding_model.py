# orig pix2pix model

import torch
import torch.nn as nn
from wavelet_thresholds.bayes import BayesThreshold
from wavelet_thresholds.msethreshold import MyThresh
from wavelet_thresholds.rigrsure import RigrsureThreshold
from wavelet_thresholds.transformershrink import TransformerWaveletThreshold
from .base_model import BaseModel
from . import networks
from utils import Denormalize, load_model, Normalize, cosim_loss,  load_model_1, spectral_loss
import numpy as np
import torch.nn.functional as F

class Pix2pixSoftWaveletThresholdingModel(BaseModel):
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
        self.metrics = ['gen_Acc', 'og_Acc', 'lp_Acc' , 'lp_adv_Acc' ]
        self.loss_names = self.metrics.copy()
        self.backward_fns = []

        self.netG=None
        if not opt.no_regenerator:
            # define networks (both generator and discriminator)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,instance_d=self.opt.instance_d)
       
        if opt.transform_type =="mse":
            netW = MyThresh(opt.wvlt, "symmetric" , opt.decomposition_level ,opt.threshold_strategy).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
            
        elif opt.transform_type =="bayes":
            netW = BayesThreshold(opt.wvlt, "symmetric" , opt.decomposition_level ,opt.threshold_strategy).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
            
        elif opt.transform_type =="transformershrink":
            netW = TransformerWaveletThreshold(opt.input_nc, opt.decomposition_level, opt.emb_size, opt.wvlt,"symmetric" , opt.num_heads, opt.ffn_hidden, 0 , opt.num_layers).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
        
        elif opt.transform_type =="rigsure":
            netW = RigrsureThreshold(opt.wvlt, "symmetric" , opt.decomposition_level ,opt.threshold_strategy).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
        
        else:
            self.netW = None

        self.sigmoid = torch.nn.Sigmoid()
        self.opt = opt
        
        self.norm= Normalize(mean=[0.5], std=[0.5]).to(self.device)
        self.denorm = Denormalize(mean=[0.5], std=[0.5]).to(self.device)
        
        opt.device = self.device

        # G_cosim_wc W_wc_kl  means cosim and wc loss is defined on output of regenerator and kl , wc loss is defined on output of wavelet denoiser
        net_losses = opt.reg.split(":")             #give loss names seperated by _ e.g cosim_kl_wc  is 3 losses cosim, kl and wc
        for losses in net_losses:
            losses = losses.split("_")
            if losses[0] =="G":
                print("loss function on output of regenerator network defnined")
                assert self.netG is not None
                self.backward_fns.append(self.backward_G)
                losses = losses[1:]
                for loss in losses:
                    self.loss_names.append("G_"+loss)
            elif losses[0] =="W":
                print("loss function on output of wavelet denoiser network defnined")
                assert self.netW is not None
                self.backward_fns.append(self.backward_W)
                losses = losses[1:]
                for loss in losses:
                    self.loss_names.append("W_"+loss)
            else:
                print("unknown network ",losses[0])
                raise Exception("error in loss functions parameters")

        print(self.loss_names)
        if self.opt.instance_d==True:
            self.loss_names.append('G_instance')
        
        if opt.surrogate_training:
            self.netT = load_model_1(opt.surrogate_teacher_model,opt.surrogate_load_path,opt.device, load_as_D=False, input_channels=opt.input_nc)
        else:
            self.netT = load_model(opt, load_as_D=False)

        self.netT.eval()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterioncosim = cosim_loss
            self.criterionspectral = spectral_loss(opt)
            self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
            self.crossentropy = nn.CrossEntropyLoss()

            if self.netG:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.model_names.append("G")
                self.optimizers.append(self.optimizer_G)

            if self.netW and sum(p.numel() for p in self.netW .parameters()) > 0:
                self.optimizer_W = torch.optim.Adam(self.netW.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.model_names.append("W")
                self.optimizers.append(self.optimizer_W)
            #else:
            #   print("no learnable module defined netW or netG")
        
          
    def set_input(self, input, attack = None):  # epoch is not used, but on removing this argument older client code will break , so keeping it with default value
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
    
        self.real_clean = input[0]                 #real_clean :> originalimage 
        #load clear and adversarial image
        self.labels = input[-1].to(self.device)

        self.real_adv = None
        if len(input)> 2:
            self.real_adv = input[1]
            self.real_adv = self.real_adv.to(self.device,dtype=torch.float)
        elif attack is not None:
            self.real_adv = attack(self.real_clean, self.labels).to(self.device,dtype=torch.float)

        self.real_clean = self.real_clean.to(self.device, dtype=torch.float)  

    def forward(self):
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        #self.real_wv_clean = self.norm(self.real_wv_clean)     
        self.real_clean = self.norm(self.real_clean)   #normalized images are fed to wavelet noise remover ref : eq.18 https://core.ac.uk/download/pdf/147900624.pdf 
        

        #move to cuda device
        if self.netW:
            self.real_wv_clean , self.t_clean_norm, self.w_clean_norm = self.netW(self.real_clean)
        else:
            self.real_wv_clean  = self.real_clean
        
        #self.real_wv_clean = self.real_clean   #remove this line later once you uncomment above code
        

        #do similar operation of adversarial image as that of clean image if it is not null
        if self.real_adv is not None:
            self.real_adv = self.norm(self.real_adv)
            
            if self.netW:     
                self.real_wv_adv , self.t_adv_norm, self.w_adv_norm = self.netW(self.real_adv, self.real_clean)
            else:
                self.real_wv_adv = self.real_adv

            self.real_wv_adv= self.norm(self.real_wv_adv)
            self.real_adv = self.norm(self.real_adv)


        if self.netG:
            self.fake_clean = self.netG(self.real_wv_clean)  # G(A)
        
            if self.real_adv is not None:
                self.fake_adv = self.netG(self.real_wv_adv)
      
        
    def backward_W(self):

        self.loss_W = None
        
        for loss in self.loss_names :
            
            if loss in self.metrics or loss[0]=="G":
                continue
            
            if loss == "W_cosim":
                
                teacher_pred_real_clean = self.netT(self.denorm(self.real_clean))
                teacher_pred_wv_clean = self.netT(self.denorm(self.real_wv_clean))
                self.loss_W_cosim = self.criterioncosim(teacher_pred_wv_clean ,teacher_pred_real_clean.detach())
                loss_w = self.loss_W_cosim
            
            elif loss =="W_kl":

                teacher_pred_real_clean = self.netT(self.denorm(self.real_clean))
                teacher_pred_wv_clean = self.netT(self.denorm(self.real_wv_clean))
                teacher_pred_wv_adv = self.netT(self.denorm(self.real_wv_adv))
                self.loss_W_kl = self.kl_criterion(F.log_softmax(teacher_pred_wv_adv/self.opt.kl_temp, dim=1),
                                F.softmax(teacher_pred_wv_clean.detach()/self.opt.kl_temp, dim=1)) * (self.opt.kl_temp * self.opt.kl_temp)
                loss_w = self.loss_W_kl
            
            elif loss=="W_wc":
                self.loss_W_wc = self.criterionL1(self.real_wv_clean, self.real_clean.detach())
                loss_w = self.loss_W_wc
                
            elif loss=="W_wcadv":
                self.loss_W_wcadv = self.criterionL1(self.real_wv_adv, self.real_clean.detach())
                loss_w = self.loss_W_wcadv
            
            elif loss == "W_reg":
                self.loss_W_reg = self.opt.reg_lambda * (self.t_clean_norm + self.t_adv_norm)
                loss_w = self.loss_W_reg
            
            else:
                print("unknown loss function ", loss)
                raise Exception("unknown loss function")
            
            if self.loss_W is None :
                self.loss_W =loss_w
            else:
                self.loss_W = self.loss_W + loss_w

        self.loss_W.backward(retain_graph=True)
        
    def backward_G(self):
        self.loss_G =None

        for loss in self.loss_names:
            if loss in self.metrics or loss[0]=="W":
                continue

            if loss == "G_cosim":
                teacher_pred_real_clean = self.netT(self.denorm(self.real_clean))
                teacher_pred_fake_clean = self.netT(self.denorm(self.fake_clean))
                self.loss_G_cosim = self.criterioncosim(teacher_pred_fake_clean ,teacher_pred_real_clean.detach())
                loss_g = self.loss_G_cosim
            
            elif loss =="G_kl":
                teacher_pred_real_clean = self.netT(self.denorm(self.real_clean))
                teacher_pred_fake_clean = self.netT(self.denorm(self.fake_clean))
                teacher_pred_fake_adv = self.netT(self.denorm(self.fake_adv))
                self.loss_G_kl = self.kl_criterion(F.log_softmax(teacher_pred_fake_adv/self.opt.kl_temp, dim=1),
                                F.softmax(teacher_pred_fake_clean.detach()/self.opt.kl_temp, dim=1)) * (self.opt.kl_temp * self.opt.kl_temp)
                loss_g = self.loss_G_kl
            
            elif loss=="G_wc":
                self.loss_G_wc = self.criterionL1(self.fake_clean, self.real_clean.detach())
                loss_g = self.loss_G_wc
            
            else:
                print("unknown loss function ", loss)
                raise Exception("unknown loss function")
            
            if self.loss_G is None:
                self.loss_G = loss_g
            else:
                self.loss_G = self.loss_G + loss_g

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        for optimizer in self.optimizers:
            optimizer.zero_grad()        # set G's gradients to zero
        for backward_fn in self.backward_fns:
            backward_fn()                 # call all backward functions
        for optimizer in self.optimizers:
            optimizer.step()              # optimize

    def init_metrics(self):

        self.lp_metrics = {'correct':0, 'total':1}
        self.gen_metrics = {'correct':0, 'total':1}
        self.og_metrics = {'correct':0, 'total':1}   
        self.lp_adv_metrics = {'correct':0, 'total':1}   
        self.gen_adv_metrics = {'correct':0, 'total':1}    


    def eval_teacher(self, test=False):
               
        with torch.no_grad():

            if self.netG:
                # Accuracy on Reconstructed  
                self.teach_fake_clean_pred = self.netT(self.denorm(self.fake_clean))[:self.opt.batch_size]
                _, pred = torch.max(self.teach_fake_clean_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.gen_metrics['correct'] += correct
                self.gen_metrics['total']   += self.teach_fake_clean_pred.size(0)
                self.loss_gen_Acc = (self.gen_metrics['correct']/self.gen_metrics['total']) * 100.

                if self.real_adv is not None:
                    # Accuracy on Low Passv adversarial
                    self.teach_fake_adv_pred = self.netT(self.denorm(self.fake_adv))[:self.opt.batch_size]
                    _, pred = torch.max(self.teach_fake_adv_pred, 1)
                    correct = (pred == self.labels).float().sum(0).item()
                    self.gen_adv_metrics['correct'] += correct
                    self.gen_adv_metrics['total']   += self.teach_fake_adv_pred.size(0)
                    self.loss_gen_adv_Acc = (self.gen_adv_metrics['correct']/self.gen_adv_metrics['total']) * 100.



            else:
                self.loss_gen_Acc =0 

            if self.netW:
                
                # Accuracy on Low Pass
                self.teach_real_wv_clean_pred = self.netT(self.denorm(self.real_wv_clean))[:self.opt.batch_size]
                _, pred = torch.max(self.teach_real_wv_clean_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.lp_metrics['correct'] += correct
                self.lp_metrics['total']   += self.teach_real_wv_clean_pred.size(0)
                self.loss_lp_Acc = (self.lp_metrics['correct']/self.lp_metrics['total']) * 100.

                if self.real_adv is not None:
                    
                    # Accuracy on Low Passv adversarial
                    self.teach_real_wv_adv_pred = self.netT(self.denorm(self.real_wv_adv))[:self.opt.batch_size]
                    _, pred = torch.max(self.teach_real_wv_adv_pred, 1)
                    correct = (pred == self.labels).float().sum(0).item()
                    self.lp_adv_metrics['correct'] += correct
                    self.lp_adv_metrics['total']   += self.teach_real_wv_adv_pred.size(0)
                    self.loss_lp_adv_Acc = (self.lp_adv_metrics['correct']/self.lp_adv_metrics['total']) * 100.


            else:
                self.loss_lp_Acc =0 
            
            ## Accuracy on clean
            self.teach_real_clean_pred = self.netT(self.denorm(self.real_clean))[:self.opt.batch_size]
            _, pred = torch.max(self.teach_real_clean_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_clean_pred.size(0)  
            self.loss_og_Acc = (self.og_metrics['correct']/self.og_metrics['total']) * 100.
            
