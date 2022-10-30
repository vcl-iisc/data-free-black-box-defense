from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        
#         parser.add_argument('--teacher_model', type=str, default='resnet18', help='teacher model')
#         parser.add_argument('--load_path', type=str, default='checkpoint_cifar10/baseline_resnet18.pth', help='teacher model path')
#         parser.add_argument('--device', type=int, default=0, help='GPU to be used')
        parser.add_argument('--num_classes', type=int, default=10)
        # parser.add_argument('--sigma', type=float, default=0.0)

        # parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
        # parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
        parser.add_argument("--sigma", type=float, help="noise hyperparameter")
        parser.add_argument("--outfile", type=str, help="output file")
        parser.add_argument("--batch", type=int, default=1, help="batch size")
        parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
        parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
        parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
        parser.add_argument("--N0", type=int, default=100)
        parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
        parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
        parser.add_argument('--denoiser', type=str, default='',
                            help='Path to a denoiser to attached before classifier during certificaiton.')
        parser.add_argument('--azure_datastore_path', type=str, default='',
                            help='Path to imagenet on azure')
        parser.add_argument('--philly_imagenet_path', type=str, default='',
                            help='Path to imagenet on philly')
            
        
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        

        ## arugments related to surrogate model training
        parser.add_argument('--use_surrogate_teacher', type=bool, default='False',
                            help='if set to true, teacher model is surrogate model')


        #black box victim model parameters
        parser.add_argument('--victim_model', type=str,
                            help="victim model name" , default=None)
        
        parser.add_argument('--victim_model_path', type=str, 
                            help='victim model path', default= None)

        #black box victim model parameters
        parser.add_argument('--defender_surrogate_model', type=str,
                            help="defender surrogate model" , default=None)
        
        parser.add_argument('--defender_surrogate_load_path', type=str, 
                            help='defender surrogate model path', default= None)




        parser.add_argument('--save_results_file', type=str, 
                            help='victim model path', default= "results6.txt")      

        parser.add_argument("--weight" , default=None, help =  "thresholding weight used by WNR")
                                               
        self.isTrain = False
        return parser
