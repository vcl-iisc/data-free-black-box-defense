# MNIST Test_model.py

from .base_model import BaseModel
from . import networks, resnet
from utils import get_freq, get_comp_wv_batch, get_LL_batch, visualize_inputs, get_k_compfrom_HF_batch, Denormalize, gaussian_noise, get_LL_batch_mnist, get_HFtop_compfrom_HF_batch_mnist, get_LL_mnist_torch
import torch
import torchattacks
import numpy as np
from DA.utils import load_model, Normalize, Trades_adv
import os.path as osp
import torch.nn as nn
import cv2
from skimage.measure import compare_ssim



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
        parser.add_argument('--model_to_attack',type=str,default='Final_model')
        parser.add_argument('--adv_samples',type=str,default='KL')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.radius = opt.lp_rad
        self.opt = opt
        print('Coefficient Percent :',self.opt.coefficients_percent)
        print('decomposition_level :',self.opt.decomposition_level)
        opt.device = self.device
#         self.netT = load_model(opt)
#         self.netT.eval()
#         self.ssim_score = []
#         self.avg_ssim = 0
        if self.opt.SaveTestImages:
            self.data_list = [None]*(60000)
            self.adv_data_list = [None]*(60000)
            self.c = 0
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
#         print(self.netG.module.get_device())
#         exit()
#         mod = [self.netG, torch.nn.DataParallel(self.netT)]
        self.norm = Normalize(mean=[0.5], std=[0.5]).to(self.device)
        self.denorm = Normalize(mean=[-0.5/0.5], std=[1/0.5]).to(self.device)
        opt.output_dir = osp.join('/media2/inder/SHOT/digit/ckps_digits/seed' + str(opt.seed), opt.adapt_mode)
        opt.class_num = 10
        self.netF, self.netB, self.netC = load_model(opt)
        self.netF.eval()
        self.netB.eval()
        self.netC.eval()
        self.ll_transform = get_LL_mnist_torch(lev=self.opt.decomposition_level).float().cuda()
        if opt.model_to_attack=='Final_model' :
            print('Attacking Final_model')
            self.Final_model = nn.Sequential(self.ll_transform, self.norm, self.netG.module, self.denorm, self.netF,self.netB,self.netC)
#         Final_model = nn.Sequential(self.norm, self.netG.module, self.denorm, self.netF,self.netB,self.netC)
        if opt.model_to_attack=='Shot_model' :
            print('Attacking Only Shot Model')
            if self.opt.adv_samples=='MMD':
                self.Final_model = nn.Sequential(self.netF, self.netB)
            else : 
                self.Final_model = nn.Sequential(self.netF,self.netB,self.netC)            


        if opt.attack == 'PGD':        
#             self.attack = torchattacks.PGD(self.netT, eps=8/255, alpha=2/255, steps=20)
            print('Attack using PGD')
            self.attack = torchattacks.PGD(self.Final_model, eps=0.3, alpha=0.01, steps=100)     #test time parameters
#             self.attack = torchattacks.PGD(Final_model, eps=0.3, alpha=0.01, steps=40)     #train time parameters
        if opt.attack == 'AA':
            self.attack = torchattacks.AutoAttack(self.Final_model, eps=0.3, n_classes=10, version='standard')


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
            self.labels = input[1].to(self.device)
            
        self.real_B = self.real_B.to(self.device)

            
        if self.opt.attack=='Trades_KL':
            print(f'using {self.opt.attack}') 
            self.real_A = Trades_adv(self.real_B,100,0.3,'l_inf',0.01,self.Final_model,self.opt.adv_samples,self.device)
        else :
            self.real_A = self.attack(self.real_B, self.labels)
#         self.real_A = self.real_B
#             self.real_A = gaussian_noise(self.real_B)    

        self.real_A = self.real_A.to(self.device)

        if self.opt.transform_type=='lp':
            self.real_A_lp = get_freq(self.real_A, self.opt.lp_rad)[0].float().to(self.device)
            self.real_B_lp = get_freq(self.real_B, self.opt.lp_rad)[0].float().to(self.device)
        elif self.opt.transform_type=='wv':
            self.real_A_lp  = get_comp_wv_batch(self.real_A, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
            self.real_B_lp  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
        elif self.opt.transform_type=='wv_custom':
#             self.real_B_lp  = get_HFtop_compfrom_HF_batch_mnist(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
#             self.real_A_lp  = get_HFtop_compfrom_HF_batch_mnist(self.real_A, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_B_lp  = get_k_compfrom_HF_batch(self.real_B, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_A_lp  = get_k_compfrom_HF_batch(self.real_A, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
        elif self.opt.transform_type=='ll':
            self.real_A_lp = get_LL_batch_mnist(self.real_A,level=self.opt.decomposition_level).float().to(self.device)
            self.real_B_lp = get_LL_batch_mnist(self.real_B, level=self.opt.decomposition_level).float().to(self.device)
        
#         print(np.transpose(self.real_B[0].detach().cpu().numpy(), (1,2,0)).shape)
#         grayA = cv2.cvtColor(np.transpose(self.real_B[0].detach().cpu().numpy(), (1,2,0)), cv2.COLOR_BGR2GRAY)
#         grayB = cv2.cvtColor(np.transpose(self.real_B_lp[0].detach().cpu().numpy(), (1,2,0)), cv2.COLOR_BGR2GRAY) 
#         (score, diff) = compare_ssim(self.real_B[0][0].detach().cpu().numpy(), self.real_B_lp[0][0].detach().cpu().numpy(), full=True)
#         self.ssim_score.append(score)
#         self.avg_ssim = self.avg_ssim + score


#         self.real_A_lp = self.attack(self.real_B_lp, self.labels)
#         self.real_A_lp = get_LL_batch_mnist(self.real_A_lp,level=self.opt.decomposition_level).float().to(self.device)
#         self.real_B_lp = get_LL_batch_mnist(self.real_B, level=self.opt.decomposition_level).float().to(self.device)        


        

    def forward(self):
        """Run forward pass."""
        self.real_B_lp_normalised = self.norm(self.real_B_lp)
        self.real_A_lp_normalised = self.norm(self.real_A_lp)
        self.fake_B = self.netG(self.real_B_lp_normalised)  # G(real)
        self.fake_A = self.netG(self.real_A_lp_normalised)  # G(real)
        self.fake_B = self.denorm(self.fake_B)
        self.fake_A = self.denorm(self.fake_A)

#         visualize_inputs(self.real_B, 'real_B')
#         visualize_inputs(self.real_A, 'real_A')
#         delta = torch.max(torch.abs(self.real_B-self.real_A))
#         print(delta)
#         visualize_inputs(self.real_B_lp, 'real_B_lp')
#         visualize_inputs(self.real_A_lp, 'real_A_lp')
#         visualize_inputs(self.fake_B, 'fake_b')
#         visualize_inputs(self.fake_A, 'fake_a')
#         exit()
        
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

    def eval_teacher(self):
        
        
        with torch.no_grad():
            # Accuracy on Reconstructed Clean image  
            self.netF.eval()
            self.netB.eval()
            self.netC.eval()
 
            ## Accuracy on clean image
            self.teach_real_B_pred = self.netC(self.netB(self.netF(self.real_B)))
            _, pred = torch.max(self.teach_real_B_pred, 1)
#             self.labels = pred
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)

            self.teach_real_pred = self.netC(self.netB(self.netF(self.fake_B)))
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_adv_metrics['correct'] += correct
            self.gen_adv_metrics['total']   += self.teach_real_pred.size(0)
            
            # Accuracy on Reconstructed Perturbed image    
            self.teach_real_pred = self.netC(self.netB(self.netF(self.fake_A)))
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_real_pred.size(0)


            # Accuracy on Low Pass Perturbed image
            self.teach_real_A_lp_pred = self.netC(self.netB(self.netF(self.real_A_lp)))
            _, pred = torch.max(self.teach_real_A_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_lp_pred.size(0)
            
            # Accuracy on Low Pass Clean image
            self.teach_real_B_lp_pred = self.netC(self.netB(self.netF(self.real_B_lp)))
            _, pred = torch.max(self.teach_real_B_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_adv_metrics['correct'] += correct
            self.lp_adv_metrics['total']   += self.teach_real_B_lp_pred.size(0)

            
            ## Accuracy on Perturbed image
            self.teach_real_A_pred = self.netC(self.netB(self.netF(self.real_A)))
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_A_pred.size(0)

    
    