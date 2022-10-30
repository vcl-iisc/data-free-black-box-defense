from .base_model import BaseModel
from . import networks, resnet
from utils import get_freq, load_model, Normalize, get_comp_wv_batch, get_LL_batch, get_ll_orig_batch, get_HFmasked_ll_batch, get_ll_orig_inverse_batch, visualize_inputs, get_ll_orig_inverse_batch8, get_HFtop_comp_batch ,get_HFtop_comp_inv_batch, get_HFtop_comp_lev_inv_batch, get_HFtop_comp_lev_batch, get_ll_orig_inverse_torch_batch, get_HFtop_comp_inv_torch_batch
import torch
import torchattacks
import numpy as np
import torchvision.transforms as transforms


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
        self.model_names = ['G1' + opt.model_suffix, 'G2' + opt.model_suffix]  # only generator is needed.
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_16', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,instance_d=self.opt.instance_d)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,instance_d=self.opt.instance_d)
        self.radius = opt.lp_rad
        self.opt = opt
        self.adv = opt.adv
        print('LP RADIUS :',self.radius)
        opt.device = self.device
        self.netT = load_model(opt)
        self.netT.eval()
        if self.opt.SaveTestImages:
            self.data_list = [None]*(50000)
            self.data_list_orig = [None]*(50000)
            self.data_list_ll = [None]*(50000)
            self.data_list_ll_input = [None]*(50000)
            if self.adv==True:
                self.adv_data_list = [None]*(50000)
                self.adv_data_list_orig = [None]*(50000)
                self.adv_data_list_ll = [None]*(50000)
                self.adv_data_list_ll_input = [None]*(50000)                
#             self.data_list_orig_l1 = [None]*(50000)
#             self.data_list_ll_l1 = [None]*(50000)
            self.c = 0
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG1' + opt.model_suffix, self.netG1)  # store netG in self.
        setattr(self, 'netG2' + opt.model_suffix, self.netG2)  # store netG in self.
        
        if opt.attack == 'PGD':        
            self.attack = torchattacks.PGD(self.netT, eps=8/255, alpha=2/255, steps=20)
        if opt.attack == 'AA':
            self.attack = torchattacks.AutoAttack(self.netT, eps=8/255, n_classes=10, version='standard')
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)       
        self.denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),])
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
# #         self.real = input['A'].to(self.device)
# #         self.image_paths = input['A_paths']
        if self.opt.SaveTestImages:
            self.real_B_orig = (input[0].unsqueeze(0)).to(self.device)
            self.labels = (torch.tensor(input[1]).unsqueeze(0)).to(self.device)  
        else:    
            self.real_B_orig = input[0].to(self.device)
            self.labels = input[1].to(self.device) 
#             self.real_B_orig = self.attack(self.real_B_orig, self.labels)
           
#         self.real_A1 = get_HFmasked_ll_batch(self.real_B_orig, lev=1).float().to(self.device)/2  # ll' l1
        self.real_A1 = get_HFtop_comp_batch(self.real_B_orig, lev=1, keep = 15/100).float().to(self.device)/2  # ll' l1


#         if self.opt.transform_type=='lp':
#             self.real_A_lp = get_freq(self.real_A, self.opt.lp_rad)[0].float().to(self.device)
#             self.real_B_lp = get_freq(self.real_B, self.opt.lp_rad)[0].float().to(self.device)
#         elif self.opt.transform_type=='wv':
#             self.real_A_lp  = get_comp_wv_batch(self.real_A, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
#             self.real_B_lp  = get_comp_wv_batch(self.real_B, lev=self.opt.decomposition_level, keep = self.opt.coefficients_percent/100).float().to(self.device)  
#         elif self.opt.transform_type=='ll':
#             self.real_A_lp = get_LL_batch(self.real_A).float().to(self.device)
#             self.real_B_lp = get_LL_batch(self.real_B).float().to(self.device)

#         self.real_B_orig = input[0].to(self.device)                 #real_B :> original image
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_B = get_ll_orig_batch(self.real_B_orig,lev=1).float().to(self.device)/2 ## ll of level 1 (contain all level2 features)
    #         print(torch.min(self.real_B), torch.max(self.real_B))
#         self.real_A = get_HFtop_comp_batch(self.real_B_orig, lev=1, keep = self.opt.coefficients_percent/100).float().to(self.device)/2## ll of level 1 (contains only ll level 2)
#         self.real_A = get_HFtop_comp_batch(self.real_B_orig, lev=1, keep = self.opt.coefficients_percent/100).float().to(self.device)/2## ll of level 1 (contains only ll level 2)
#         self.real_A = get_HFtop_comp_lev_batch(self.real_B_orig, lev=1, keep = 0.15).float().to(self.device)/2

#         print(torch.min(self.real_A), torch.max(self.real_A))
#         self.real_B_orig = self.norm(self.real_B_orig)
#         print(torch.min(self.real_B_orig), torch.max(self.real_B_orig))
        self.real_A1 = self.norm(self.real_A1)     
    #         print(torch.min(self.real_A), torch.max(self.real_A))
        self.real_B = self.norm(self.real_B)
        #         print(torch.min(self.real_B), torch.max(self.real_B))

        if self.adv==True:
            self.adv_B_orig = self.attack(self.real_B_orig, self.labels)
            self.adv_B = get_ll_orig_batch(self.adv_B_orig,lev=1).float().to(self.device)/2 ## ll of level 1 (contain all level2 features)
#             self.adv_A = get_HFtop_comp_batch(self.adv_B_orig, lev=1, keep = self.opt.coefficients_percent/100).float().to(self.device)/2## ll of level 1 (contains only ll level 2)
#             self.adv_A = get_HFtop_comp_batch(self.adv_B_orig, lev=1).float().to(self.device)/2## ll of level 1 (contains only ll level 2)
            self.adv_A = get_HFtop_comp_lev_batch(self.adv_B_orig, lev=1, keep = 0.15).float().to(self.device)/2
            self.adv_A = self.norm(self.adv_A)     
            self.adv_B = self.norm(self.adv_B)


#         self.labels = input[1].to(self.device)
        

    def forward(self):
        """Run forward pass."""
#         self.fake = self.netG(self.real)  # G(real)
#         self.real_B_lp_normalised = self.norm(self.real_B_lp)
#         self.real_A_lp_normalised = self.norm(self.real_A_lp)
#         self.fake_B = self.netG(self.real_B)  # G(real)
        self.fake_A1 = self.netG1(self.real_A1)  # G(real)
        self.real_A2 = get_HFtop_comp_inv_torch_batch(self.opt,self.real_B_orig,self.denorm(self.fake_A1)*2,keep=0.15,lev=1).float().to(self.device) 
#         self.real_A2 = get_ll_orig_inverse_torch_batch(self.opt,self.denorm(self.fake_A1)*2).float().to(self.device) 
        self.fake_A2 = self.netG2(self.real_A2)  # G(A)

        self.real_B = get_ll_orig_inverse_batch(self.denorm(self.real_B)*2).float().to(self.device)
#         self.real_B = get_HFtop_comp_lev_inv_batch(self.real_B_orig,self.denorm(self.real_B)*2,keep=0.15,lev=1).float().to(self.device)
        self.fake_A1 = get_ll_orig_inverse_batch(self.denorm(self.fake_A1)*2).float().to(self.device)
#         visualize_inputs(self.fake_A, 'fake_A')
#         print('fake_A: ', torch.min(self.fake_A2),torch.max(self.fake_A2))        
#         self.fake_A = get_HFtop_comp_lev_inv_batch(self.real_B_orig,self.denorm(self.fake_A)*2,keep=0.15,lev=1).float().to(self.device)
#         visualize_inputs(self.fake_A, 'fake_A_2')
#         print('fake_A: ', torch.min(self.fake_A),torch.max(self.fake_A))
        self.real_A1 = get_ll_orig_inverse_batch(self.denorm(self.real_A1)*2).float().to(self.device)
#         self.real_A1 = get_HFtop_comp_lev_inv_batch(self.real_B_orig,self.denorm(self.real_A1)*2,keep=0.15,lev=1).float().to(self.device)
#         exit()
        if self.adv==True:
            self.adv_fake_A = self.netG(self.adv_A)  # G(real)
            self.adv_B = get_HFtop_comp_lev_inv_batch(self.adv_B_orig,self.denorm(self.adv_B)*2,keep=0.15,lev=1).float().to(self.device)
#             self.adv_B = get_ll_orig_inverse_batch(self.denorm(self.adv_B)*2).float().to(self.device)
#             self.adv_fake_A = get_ll_orig_inverse_batch(self.denorm(self.adv_fake_A)*2).float().to(self.device)
#             self.adv_fake_A = get_HFtop_comp_inv_batch(self.adv_B_orig,self.denorm(self.adv_fake_A)*2, keep=0.15,lev=1).float().to(self.device)
#             self.adv_A = get_ll_orig_inverse_batch(self.denorm(self.adv_A)*2).float().to(self.device)
            self.adv_fake_A = get_HFtop_comp_lev_inv_batch(self.adv_B_orig,self.denorm(self.adv_fake_A)*2,keep=0.15,lev=1).float().to(self.device)
            self.adv_A = get_HFtop_comp_lev_inv_batch(self.adv_B_orig,self.denorm(self.adv_A)*2,keep=0.15,lev=1).float().to(self.device)
#         visualize_inputs(self.fake_A, 'fake_B')
#         print('fake_A: ', torch.min(self.fake_A),torch.max(self.fake_A))
#         visualize_inputs(self.real_A, 'real_A')
#         print('real_A: ', torch.min(self.real_A),torch.max(self.real_A))
#         visualize_inputs(self.real_B, 'real_B')
#         print('real_B: ', torch.min(self.real_B),torch.max(self.real_B))
#         visualize_inputs(self.real_B_orig, 'real_B_orig')
#         print('real_B_orig: ', torch.min(self.real_B_orig),torch.max(self.real_B_orig))

#         exit()

        
        if self.opt.SaveTestImages:
            self.data_list[self.c] = self.fake_A.squeeze(0)
            self.data_list_orig[self.c] = self.real_B_orig.squeeze(0)
            self.data_list_ll[self.c] = self.real_B.squeeze(0)
            self.data_list_ll_input[self.c] = self.real_A.squeeze(0)
            if self.adv==True:
                self.adv_data_list[self.c] = self.adv_fake_A.squeeze(0)
                self.adv_data_list_orig[self.c] = self.adv_B_orig.squeeze(0)
                self.adv_data_list_ll[self.c] = self.adv_B.squeeze(0)
                self.adv_data_list_ll_input[self.c] = self.adv_A.squeeze(0)
            self.c = self.c+1
    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def init_metrics(self):
        self.lp_adv_metrics = {'correct':0, 'total':1}
        self.lp_metrics = {'correct':0, 'total':0}
        self.gen_adv_metrics = {'correct':0, 'total':0}
        self.gen_metrics = {'correct':0, 'total':0}
        self.og_metrics = {'correct':0, 'total':0}
        self.adv_metrics = {'correct':0, 'total':0}
        if self.adv==False:
            self.gen_adv_metrics = {'correct':0, 'total':1}
            self.lp_adv_metrics = {'correct':0, 'total':1}


    def eval_teacher(self):
        
        
        with torch.no_grad():
            # Accuracy on Reconstructed Clean image  
            
#             if self.adv==True:              
            self.teach_real_pred = self.netT(self.denorm(self.fake_A2))
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_adv_metrics['correct'] += correct
            self.gen_adv_metrics['total']   += self.teach_real_pred.size(0)

#             # Accuracy on Low Pass Clean image
#             self.teach_real_B_lp_pred = self.netT(self.adv_B)
#             _, pred = torch.max(self.teach_real_B_lp_pred, 1)
#             correct = (pred == self.labels).float().sum(0).item()
#             self.lp_adv_metrics['correct'] += correct
#             self.lp_adv_metrics['total']   += self.teach_real_B_lp_pred.size(0)
            
            # Accuracy on Reconstructed Perturbed image    
            self.teach_real_pred = self.netT(self.fake_A1)
            _, pred = torch.max(self.teach_real_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.gen_metrics['correct'] += correct
            self.gen_metrics['total']   += self.teach_real_pred.size(0)


            # Accuracy on Low Pass Perturbed image
            self.teach_real_A_lp_pred = self.netT(self.real_B)
            _, pred = torch.max(self.teach_real_A_lp_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.lp_metrics['correct'] += correct
            self.lp_metrics['total']   += self.teach_real_A_lp_pred.size(0)
            
            ## Accuracy on clean image
            self.teach_real_B_pred = self.netT(self.real_B_orig)
            _, pred = torch.max(self.teach_real_B_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.og_metrics['correct'] += correct
            self.og_metrics['total']   += self.teach_real_B_pred.size(0)
            
            ## Accuracy on Perturbed image
            self.teach_real_A_pred = self.netT(self.real_A1)
            _, pred = torch.max(self.teach_real_A_pred, 1)
            correct = (pred == self.labels).float().sum(0).item()
            self.adv_metrics['correct'] += correct
            self.adv_metrics['total']   += self.teach_real_A_pred.size(0)

    
    