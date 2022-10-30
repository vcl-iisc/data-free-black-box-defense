# DA Office Test_model.pyin

import os

from adversarial_attacks import get_test_attack
from wavelet_thresholds.bayes import BayesThreshold
from wavelet_thresholds.rigrsure import RigrsureThreshold
from wavelet_thresholds.transformershrink import TransformerWaveletThreshold

from .base_model import BaseModel
from . import networks, resnet
from utils import MinMaxNormalization, get_freq, get_comp_wv_batch, get_LL_batch, load_model_1, visualize_inputs, get_k_compfrom_HF_batch, Normalize,  Denormalize, gaussian_noise, get_LL_batch_mnist, get_HFtop_compfrom_HF_batch_mnist, get_LL_torch, load_model, wvlt_transform
import torch
import torchattacks
import numpy as np
from DA.utils import load_model_office
import os.path as osp
import torch.nn as nn
import cv2
from skimage.metrics import structural_similarity as compare_ssim


class TestWaveletThresholdingModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #assert(not opt.isTrain)   #TODO not sure why this line is here. temporarily commented out
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = []  # only generator is needed.
        
        self.netG=None
        if not opt.no_regenerator:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.model_names.append("G")

        if opt.transform_type =="bayes":
            netW = BayesThreshold(opt.wvlt, "symmetric" , opt.decomposition_level ,opt.threshold_strategy).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
            
        elif opt.transform_type =="transformershrink":
            netW = TransformerWaveletThreshold(opt.input_nc, opt.decomposition_level, opt.emb_size, opt.wvlt,"symmetric" , opt.num_heads, opt.ffn_hidden, 0 , opt.num_layers).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
            self.model_names.append("W")
        
        elif opt.transform_type =="rigsure":
            netW = RigrsureThreshold(opt.wvlt, "symmetric" , opt.decomposition_level ,opt.threshold_strategy).to(self.device)
            self.netW = torch.nn.DataParallel(netW, self.gpu_ids)
        
        else:
            self.netW = None

    
            
            
        self.radius = opt.lp_rad
        self.opt = opt

        print('decomposition_level :',self.opt.decomposition_level)
        opt.device = self.device

        self.ssim_score = []
        self.avg_ssim = 0
        if self.opt.SaveTestImages:
            self.data_list = [None]*(50000)
            self.adv_data_list = [None]*(50000)
            self.c = 0
        
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        self.denorm = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        
        if opt.use_surrogate_teacher:
            print("input channels" ,opt.input_nc)
            self.netT = load_model_1(opt.surrogate_teacher_model, opt.surrogate_load_path, opt.device, load_as_D=False,input_channels=opt.input_nc)
        else:
            self.netT = load_model(opt, load_as_D=False)
        
        self.netT.eval()
        self.ll_transform = get_LL_torch(lev=self.opt.decomposition_level).float().cuda()

        self.attack = get_test_attack(os.path.basename(opt.dataroot) , opt.attack , self.netT) 

        
        if opt.victim_model is not None:
            print("using victim black box model {}".format(opt.victim_model))
            self.netV = load_model_1(opt.victim_model, opt.victim_model_path, opt.device, input_channels=opt.input_nc)
        else:
            print("surroagate model is victim model too...")
            self.netV = self.netT            # victim model on which testing is done is same as surrogate model
        
        
        self.netV.eval()
        
        self.coeff_percent_selection = False   # used during coeff selection testing
        
        self.defender_surrogate=None
        if opt.defender_surrogate_model is not None:
            print("using defender surrogate  model {}".format(opt.defender_surrogate_model))
            self.defender_surrogate = load_model_1(opt.defender_surrogate_model, opt.defender_surrogate_load_path, opt.device, input_channels=opt.input_nc)
            self.defender_surrogate.eval()

        

    def set_input(self, input):

        if self.opt.SaveTestImages:
            self.real_clean = (input[0].unsqueeze(0)).to(self.device)
            self.labels = (torch.tensor(input[1]).unsqueeze(0)).to(self.device)  
        else:    
            self.real_clean = input[0]   
            self.labels = input[-1].to(self.device)
            
            if len(input) == 3:   # if we get adversarial image from the dataset use that 
                self.real_adv = input[1]
            else: 
                 # otherwise compute the adversarial image.
                self.real_adv = self.attack(self.real_clean, self.labels)   # real_adv is Perturbed image
               
        self.real_clean = self.norm(self.real_clean.to(self.device))
        self.real_adv = self.norm(self.real_adv.to(self.device))

        

    def forward(self):
        """Run forward pass."""

        if self.netW:
            self.real_wv_clean , self.t_clean , self.w_clean = self.netW(self.real_clean)
            #self.real_wv_adv , self.t_adv, self.w_adv = self.netW(self.real_adv)
            #TODO uncomment above line
            self.real_wv_adv , self.t_adv, self.w_adv = self.netW(self.real_adv, self.real_clean)
            
            #self.real_wv_clean_normalised = self.norm(self.real_wv_clean)
            #self.real_wv_adv_normalised = self.norm(self.real_wv_adv)
        
        if self.netG:
            self.fake_clean = self.netG(self.real_wv_clean)  # G(real)
            self.fake_adv = self.netG(self.real_wv_adv)  # G(real)
            
            #self.fake_clean = self.denorm(self.fake_clean)
            #self.fake_adv = self.denorm(self.fake_adv)

        
        if self.opt.SaveTestImages:
            self.data_list[self.c] = self.fake_adv.squeeze(0)
            self.adv_data_list[self.c] = self.real_adv.squeeze(0)
            self.c = self.c+1

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def init_metrics(self):
        #total set to 1 to avoid divide by zero error
        self.lp_adv_metrics = {'correct':0, 'total':1}
        self.lp_metrics = {'correct':0, 'total':1}
        self.gen_adv_metrics = {'correct':0, 'total':1}
        self.gen_metrics = {'correct':0, 'total':1}
        self.og_metrics = {'correct':0, 'total':1}
        self.adv_metrics = {'correct':0, 'total':1}
        self.surrogate_og_metrics = {'correct':0, 'total':1}
        self.defender_surrogate_og_metrics = {'correct':0, 'total':1}

    def eval_teacher(self):
        with torch.no_grad():

            self.teach_real_clean_pred = self.netV(self.denorm(self.real_clean))
            _, pred = torch.max(self.teach_real_clean_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_clean_pred.size(0)
            
            
            
            ## Accuracy of attacker surrogate on clean image
            #real_clean is the originial image
            self.teach_real_clean_pred = self.netT(self.denorm(self.real_clean))
            _, s_pred = torch.max(self.teach_real_clean_pred, 1)
            
            correct = (s_pred == self.labels).float().sum(0).item()
            self.surrogate_og_metrics['correct'] += correct
            self.surrogate_og_metrics['total']   += self.teach_real_clean_pred.size(0)  


            ## Accuracy of defender surrogate on clean image
            #real_clean is the originial image
            if self.defender_surrogate is not None:
                self.teach_real_clean_pred = self.defender_surrogate(self.denorm(self.real_clean))
                _, s_pred = torch.max(self.teach_real_clean_pred, 1)
                
                correct = (s_pred == self.labels).float().sum(0).item()
                self.defender_surrogate_og_metrics['correct'] += correct
                self.defender_surrogate_og_metrics['total']   += self.teach_real_clean_pred.size(0)  


            if self.coeff_percent_selection :
                self.labels = pred      # we use black box model prediction as ground truth for coefficient selection. 
           
            if self.netG:
                #fake_B is the reconstructed image from pix2pix model
                self.teach_real_pred = self.netV(self.denorm(self.fake_clean))
                _, pred = torch.max(self.teach_real_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.gen_metrics['correct'] += correct
                self.gen_metrics['total']   += self.teach_real_pred.size(0)
                
                
                # Accuracy on Reconstructed Perturbed image 
                # in case of pix2pix fake_A is reconstructed image from perturbed image   
                self.teach_real_pred = self.netV(self.denorm(self.fake_adv))
                _, pred = torch.max(self.teach_real_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.gen_adv_metrics['correct'] += correct
                self.gen_adv_metrics['total']   += self.teach_real_pred.size(0)
            

            if self.netW:
                # Accuracy on Low Pass Perturbed image
                self.teach_real_wv_adv_pred = self.netV(self.denorm(self.real_wv_adv))
                _, pred = torch.max(self.teach_real_wv_adv_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.lp_adv_metrics['correct'] += correct
                self.lp_adv_metrics['total']   += self.teach_real_wv_adv_pred.size(0)
                


                # Accuracy on Low Pass Clean image
                self.teach_real_wv_clean_pred = self.netV(self.denorm(self.real_wv_clean))
                _, pred = torch.max(self.teach_real_wv_clean_pred, 1)
                correct = (pred == self.labels).float().sum(0).item()
                self.lp_metrics['correct'] += correct
                self.lp_metrics['total']   += self.teach_real_wv_clean_pred.size(0)
            

            ## Accuracy adversarial image
            self.teach_real_adv_pred = self.netV(self.denorm(self.real_adv))
            _, pred = torch.max(self.teach_real_adv_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_adv_pred.size(0)
            

            