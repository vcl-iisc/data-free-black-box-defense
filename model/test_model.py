# DA Office Test_model.pyin

import os

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


class TestModel(BaseModel):
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

    def __init__(self, opt, netG=None):
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
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        
        if netG:
            self.netG = netG
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
           
        self.radius = opt.lp_rad
        self.opt = opt
        print('LP RADIUS :',self.radius)
        print('Coefficient Percent :',self.opt.coefficients_percent)
        print('decomposition_level :',self.opt.decomposition_level)
        opt.device = self.device
#         self.netT = load_model(opt)
#         self.netT.eval()
        self.ssim_score = []
        self.avg_ssim = 0
        if self.opt.SaveTestImages:
            self.data_list = [None]*(50000)
            self.adv_data_list = [None]*(50000)
            self.c = 0
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
#         print(self.netG.module.get_device())
#         exit()
#         mod = [self.netG, torch.nn.DataParallel(self.netT)]
        
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        #self.denorm = Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5]).to(self.device)
        #self.denorm = MinMaxNormalization()
        self.denorm = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        
        opt.device = self.device
#         self.netF, self.netB, self.netC = load_model_office(opt)
#         self.netF.eval()
#         self.netB.eval()
#         self.netC.eval()
        if opt.use_surrogate_teacher:
            print("input channels" ,opt.input_nc)
            self.netT = load_model_1(opt.surrogate_teacher_model, opt.surrogate_load_path, opt.device, load_as_D=False,input_channels=opt.input_nc)
        else:
            self.netT = load_model(opt, load_as_D=False)
        
        self.netT.eval()
        self.ll_transform = get_LL_torch(lev=self.opt.decomposition_level).float().cuda()
#         Final_model = nn.Sequential(self.netF,self.netB,self.netC)
#         Final_model = nn.Sequential(self.ll_transform, self.norm, self.netG.module, self.denorm, self.netF,self.netB,self.netC)
        if opt.attack == 'PGD':
            if os.path.basename(opt.dataroot) =="cifar_10":
                print('Attack using PGD on cifar10 dataset')
                eps, alpha, steps = 8/255, 2/255, 20
            elif  os.path.basename(opt.dataroot) =="mnist":
                print('Attack using PGD on mnist dataset')
                eps, alpha, steps = 0.3, 0.01, 100
            elif  os.path.basename(opt.dataroot) =="fmnist":
                print('Attack using PGD on fmnist dataset')
                #eps, alpha, steps = 0.3, 0.01, 100
                eps, alpha, steps = 0.2, 0.02, 100
            elif os.path.basename(opt.dataroot) == "svhn":
                print("Attack using PGD on svhn dataset")
                eps , alpha, steps = 4/255 , 2/255,  20
            self.attack = torchattacks.PGD(self.netT, eps=eps, alpha=alpha, steps=steps)
    
        if opt.attack == 'AA':
            if os.path.basename(opt.dataroot) == 'cifar_10':
                print('Attack using AA on cifar10 dataset')
                eps = 8/255

            elif os.path.basename(opt.dataroot) == 'mnist':
                print('Attack using AA on mnist dataset')
                eps = 0.3

            elif os.path.basename(opt.dataroot) == 'fmnist':
                print('Attack using AA on fmnist dataset')
                eps = 0.2
            elif os.path.basename(opt.dataroot) == 'svhn':
                print('Attack using AA on svhn dataset')
                eps = 4/255

            self.attack = torchattacks.AutoAttack(self.netT, eps=eps, n_classes=10, version='standard')

        if opt.attack == 'BIM':
            if os.path.basename(opt.dataroot) == 'cifar_10':
                print('Attack using BIM on cifar10 dataset')
                eps, steps = 8/255, 20
                alpha = eps/steps
            elif os.path.basename(opt.dataroot) == 'mnist':
                print('Attack using BIM on mnist dataset')
                eps, alpha, steps = 0.3, 0.03, 100
            elif os.path.basename(opt.dataroot) == 'fmnist':
                print('Attack using BIM on fmnist dataset')
                eps, alpha, steps = 0.2, 0.02, 100
            elif os.path.basename(opt.dataroot) == "svhn":
                print("Attack using BIM on svhn dataset")
                eps , alpha, steps = 4/255 , 2/255,  20
            
            self.attack = torchattacks.BIM(self.netT,eps=eps, alpha=alpha, steps=steps )

        #victim model is against which testing is done.
        #since this will be a black box model, we use self.netT( surrogate model) to generate the adversarial n_samples
        if opt.victim_model is not None:
            print("using victim black box model {}".format(opt.victim_model))
            self.netV = load_model_1(opt.victim_model, opt.victim_model_path, opt.device, input_channels=opt.input_nc)
        else:
            print("surroagate model is victim model too...")
            self.netV = self.netT            # victim model on which testing is done is same as surrogate model
        
        
        self.netV.eval()
        
        self.coeff_percent_selection = False   # used during coeff selection testing

        if self.opt.transform_type is None:
            print("wavelet transformation not defined.")
        self.mask= None

        self.defender_surrogate=None
        if opt.defender_surrogate_model is not None:
            print("using defender surrogate  model {}".format(opt.defender_surrogate_model))
            self.defender_surrogate = load_model_1(opt.defender_surrogate_model, opt.defender_surrogate_load_path, opt.device, input_channels=opt.input_nc)
            self.defender_surrogate.eval()

        #if opt.victim_model is not None:
        #    self.netV = load
    """def denorm(self, x):
        batch_size = x.size(0)
        x_new = torch.zeros(x.size(), dtype=x.dtype)
        for idx in range(batch_size):
            x_new[idx] = ((x[idx] - torch.min(x[idx]))/(torch.max(x[idx])-torch.min(x[idx])))
        x_new = x_new.to(self.device)
        return x_new"""

        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
#         self.real = input['A'].to(self.device)
#         self.image_paths = input['A_paths']
        if self.opt.SaveTestImages:
            self.real_B = (input[0].unsqueeze(0)).to(self.device)
            self.labels = (torch.tensor(input[1]).unsqueeze(0)).to(self.device)  
        else:    
            self.real_B = input[0]   
            self.labels = input[-1].to(self.device)
            
            if len(input) == 3:   # if we get adversarial image from the dataset use that 
                self.real_A = input[1]
            else: 
                 # otherwise compute the adversarial image.
                self.real_A = self.attack(self.real_B, self.labels)   # real_A is Perturbed image
               
#         self.real_A = self.real_B
        #self.real_A = gaussian_noise(self.real_B)    

        self.real_B = self.real_B.to(self.device)
        self.real_A = self.real_A.to(self.device)

        if self.opt.transform_type=='lp':
            if self.mask is None:
                rows, cols = self.real_A.size(2) , self.real_A.size(3)
                mask = torch.zeros((rows, cols)).to(self.device)
                for i in range(rows):
                    for j in range(cols):
                        dis = np.sqrt((i - rows/2) ** 2 + (j - rows/2) ** 2)
                        if dis < self.opt.lp_rad:
                            mask[i, j] = 1.0
                
                self.mask = mask
            
            self.real_A_lp = get_freq(self.real_A, self.mask)[0].float()
            
            self.real_B_lp = get_freq(self.real_B, self.mask)[0].float()


        elif self.opt.transform_type=='wv':
            self.real_A_lp  = get_comp_wv_batch(self.real_A, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
            self.real_B_lp  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
        elif self.opt.transform_type in ['wv_custom' , 'rigsure', 'bayes']:
            #self.real_A_lp  = get_k_compfrom_HF_batch(self.real_A, setting=self.opt.which_k, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            #self.real_B_lp  = get_k_compfrom_HF_batch(self.real_B, setting=self.opt.which_k, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_A_lp = wvlt_transform(self.real_A, device  = self.device,lev=self.opt.decomposition_level,setting=self.opt.which_k,
                                   keep = self.opt.coefficients_percent/100,selection_strategy=self.opt.transform_type,wvlt = self.opt.wvlt , threshold_strategy = self.opt.threshold_strategy).float().to(self.device)

            self.real_B_lp = wvlt_transform(self.real_B, device  = self.device,lev=self.opt.decomposition_level,setting=self.opt.which_k,
                                   keep = self.opt.coefficients_percent/100,selection_strategy=self.opt.transform_type,wvlt = self.opt.wvlt , threshold_strategy = self.opt.threshold_strategy).float().to(self.device)

        elif self.opt.transform_type=='ll':
            #self.real_A_lp = get_LL_batch(self.real_A, level=self.opt.decomposition_level).float().to(self.device)
            #self.real_B_lp = get_LL_batch(self.real_B, level=self.opt.decomposition_level).float().to(self.device)
            self.real_A_lp = wvlt_transform(self.real_A, device  = self.device,lev=self.opt.decomposition_level,selection_strategy="ll",wvlt = self.opt.wvlt).float().to(self.device)
            self.real_B_lp = wvlt_transform(self.real_B, device  = self.device,lev=self.opt.decomposition_level,selection_strategy="ll",wvlt = self.opt.wvlt).float().to(self.device)
            
        elif self.opt.transform_type is None:
            self.real_A_lp = self.real_A
            self.real_B_lp = self.real_B
        

#         grayA = cv2.cvtColor(np.transpose(self.real_B[0].detach().cpu().numpy(), (1,2,0)), cv2.COLOR_BGR2GRAY)
#         grayB = cv2.cvtColor(np.transpose(self.real_B_lp[0].detach().cpu().numpy(), (1,2,0)), cv2.COLOR_BGR2GRAY) 
#         (score, diff) = compare_ssim(grayA, grayB, full=True)
#         self.ssim_score.append(score)
#         self.avg_ssim = self.avg_ssim + score
# #         self.real_A_lp = self.attack(self.real_B_lp, self.labels)





    def forward(self):
        """Run forward pass."""
        self.real_B_lp_normalised = self.norm(self.real_B_lp)
        self.real_A_lp_normalised = self.norm(self.real_A_lp)
        
        self.fake_B = self.netG(self.real_B_lp_normalised)  # G(real)
        self.fake_A = self.netG(self.real_A_lp_normalised)  # G(real)
        
        self.fake_B = self.denorm(self.fake_B)
        self.fake_A = self.denorm(self.fake_A)

        
#         self.fake_B = (self.fake_B - torch.min(self.fake_B))/(torch.max(self.fake_B)-torch.min(self.fake_B)) 
#         self.fake_A = (self.fake_A - torch.min(self.fake_A))/(torch.max(self.fake_A)-torch.min(self.fake_A))

        if self.opt.SaveTestImages:
            self.data_list[self.c] = self.fake_A.squeeze(0)
            self.adv_data_list[self.c] = self.real_A.squeeze(0)
            self.c = self.c+1
    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def init_metrics(self):
        self.lp_adv_metrics = {'correct':0, 'total':0}
        self.lp_metrics = {'correct':0, 'total':0}
        self.gen_adv_metrics = {'correct':0, 'total':0}
        self.gen_metrics = {'correct':0, 'total':0}
        self.og_metrics = {'correct':0, 'total':0}
        self.adv_metrics = {'correct':0, 'total':0}
        self.surrogate_og_metrics = {'correct':0, 'total':0}
        self.defender_surrogate_og_metrics = {'correct':0, 'total':0}

    def eval_teacher(self):
        with torch.no_grad():
               # Accuracy on Reconstructed Clean image  
#             self.netF.eval()
#             self.netB.eval()
#             self.netC.eval()
            #self.netV.eval()   #this line is not necessary. 
            

            ## Accuracy on clean image
            #real_B is the originial image
            self.teach_real_B_pred = self.netV(self.real_B)
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)
            
            
            
            ## Accuracy of attacker surrogate on clean image
            #real_B is the originial image
            self.teach_real_B_pred = self.netT(self.real_B)
            _, s_pred = torch.max(self.teach_real_B_pred, 1)
            
            correct = (s_pred == self.labels).float().sum(0).item()
            self.surrogate_og_metrics['correct'] += correct
            self.surrogate_og_metrics['total']   += self.teach_real_B_pred.size(0)  


            ## Accuracy of defender surrogate on clean image
            #real_B is the originial image
            if self.defender_surrogate is not None:
                self.teach_real_B_pred = self.defender_surrogate(self.real_B)
                _, s_pred = torch.max(self.teach_real_B_pred, 1)
                
                correct = (s_pred == self.labels).float().sum(0).item()
                self.defender_surrogate_og_metrics['correct'] += correct
                self.defender_surrogate_og_metrics['total']   += self.teach_real_B_pred.size(0)  


            if self.coeff_percent_selection :
                self.labels = pred      # we use black box model prediction as ground truth for coefficient selection. 
           
            #fake_B is the reconstructed image from pix2pix model
            self.teach_real_pred = self.netV(self.fake_B)
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_real_pred.size(0)
            
            
            # Accuracy on Reconstructed Perturbed image 
            # in case of pix2pix fake_A is reconstructed image from perturbed image   
            self.teach_real_pred = self.netV(self.fake_A)
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_adv_metrics['correct'] += correct
            self.gen_adv_metrics['total']   += self.teach_real_pred.size(0)
            

            # Accuracy on Low Pass Perturbed image
            self.teach_real_A_lp_pred = self.netV(self.real_A_lp)
            _, pred = torch.max(self.teach_real_A_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_adv_metrics['correct'] += correct
            self.lp_adv_metrics['total']   += self.teach_real_A_lp_pred.size(0)
            


            # Accuracy on Low Pass Clean image
            self.teach_real_B_lp_pred = self.netV(self.real_B_lp)
            _, pred = torch.max(self.teach_real_B_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_B_lp_pred.size(0)
            

            ## Accuracy adversarial image
            self.teach_real_A_pred = self.netV(self.real_A)
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_A_pred.size(0)
            

            