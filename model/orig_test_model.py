# Original Test_model.py

from .base_model import BaseModel
from . import networks, resnet
from utils import get_freq, load_model, Normalize, get_comp_wv_batch, get_LL_batch, visualize_inputs, get_k_compfrom_HF_batch, Denormalize, gaussian_noise
import torch
import torchattacks
import numpy as np


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
        print('LP RADIUS :',self.radius)
        opt.device = self.device
        self.netT = load_model(opt)
        self.netT.eval()
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
        self.denorm = Denormalize()      
        mod = [self.norm, self.netG.module, self.denorm, self.netT]
        final_model = torch.nn.Sequential(*mod)
        final_model.eval()
        if opt.attack == 'PGD':        
#             self.attack = torchattacks.PGD(self.netT, eps=8/255, alpha=2/255, steps=20)
            self.attack = torchattacks.PGD(self.netT, eps=8/255, alpha=2/255, steps=20)
        if opt.attack == 'AA':
            self.attack = torchattacks.AutoAttack(self.netT, eps=8/255, n_classes=10, version='standard')

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
        self.real_A = self.attack(self.real_B, self.labels)
#         self.real_A = self.real_B
#             self.real_A = gaussian_noise(self.real_B)    

        self.real_B = self.real_B.to(self.device)
        self.real_A = self.real_A.to(self.device)

        if self.opt.transform_type=='lp':
            self.real_A_lp = get_freq(self.real_A, self.opt.lp_rad)[0].float().to(self.device)
            self.real_B_lp = get_freq(self.real_B, self.opt.lp_rad)[0].float().to(self.device)
        elif self.opt.transform_type=='wv':
            self.real_A_lp  = get_comp_wv_batch(self.real_A, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
            self.real_B_lp  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
        elif self.opt.transform_type=='wv_custom':
            self.real_B_lp  = get_k_compfrom_HF_batch(self.real_B, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
            self.real_A_lp  = get_k_compfrom_HF_batch(self.real_A, setting='top', lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)
        elif self.opt.transform_type=='ll':
            self.real_A_lp = get_LL_batch(self.real_A,level=self.opt.decomposition_level).float().to(self.device)
            self.real_B_lp = get_LL_batch(self.real_B, level=self.opt.decomposition_level).float().to(self.device)
#             self.real_A_lp = self.attack(self.real_B_lp, self.labels)
        


        

    def forward(self):
        """Run forward pass."""
#         self.fake = self.netG(self.real)  # G(real)
        self.real_B_lp_normalised = self.norm(self.real_B)
        self.real_A_lp_normalised = self.norm(self.real_A)
#         self.fake_B = self.netG(self.real_B_lp_normalised)  # G(real)
#         self.fake_A = self.netG(self.real_A_lp_normalised)  # G(real)
#         self.real_B_lp_normalised = self.norm(self.real_B_lp)
#         self.real_A_lp_normalised = self.norm(self.real_A_lp)
        self.fake_B = self.netG(self.real_B_lp_normalised)  # G(real)
        self.fake_A = self.netG(self.real_A_lp_normalised)  # G(real)
        self.fake_B = (self.fake_B - torch.min(self.fake_B))/(torch.max(self.fake_B)-torch.min(self.fake_B)) 
        self.fake_A = (self.fake_A - torch.min(self.fake_A))/(torch.max(self.fake_A)-torch.min(self.fake_A))
#         self.fake_B = self.denorm(self.fake_B)
#         self.fake_A = self.denorm(self.fake_A)
#         self.real_B_lp_normalised = (self.real_B_lp_normalised - torch.min(self.real_B_lp_normalised))/(torch.max(self.real_B_lp_normalised)-torch.min(self.real_B_lp_normalised))
#         visualize_inputs(self.real_B[0],'train_GT')
#         print(torch.min(self.real_B[0]), torch.max(self.real_B[0]))
#         visualize_inputs(self.fake_B[0],'train_output')
#         print(torch.min(self.fake_B[0]), torch.max(self.fake_B[0]))
#         visualize_inputs(self.real_B_lp_normalised[0],'train_input')
#         print(torch.min(self.real_B_lp_normalised[0]), torch.max(self.real_B_lp_normalised[0]))
#         exit()
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
 
            self.teach_real_pred = self.netT(self.fake_B)
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_adv_metrics['correct'] += correct
            self.gen_adv_metrics['total']   += self.teach_real_pred.size(0)
            
            # Accuracy on Reconstructed Perturbed image    
            self.teach_real_pred = self.netT(self.fake_A)
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_real_pred.size(0)


            # Accuracy on Low Pass Perturbed image
            self.teach_real_A_lp_pred = self.netT(self.real_A_lp)
            _, pred = torch.max(self.teach_real_A_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_lp_pred.size(0)
            
            # Accuracy on Low Pass Clean image
            self.teach_real_B_lp_pred = self.netT(self.real_B_lp)
            _, pred = torch.max(self.teach_real_B_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_adv_metrics['correct'] += correct
            self.lp_adv_metrics['total']   += self.teach_real_B_lp_pred.size(0)


            ## Accuracy on clean image
            self.teach_real_B_pred = self.netT(self.real_B)
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)
            
            ## Accuracy on Perturbed image
            self.teach_real_A_pred = self.netT(self.real_A)
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_A_pred.size(0)

    
    