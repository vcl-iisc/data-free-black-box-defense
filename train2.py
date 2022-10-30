#Original Train.py

"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model, find_model_using_name, gan
from util.visualizer import Visualizer
from utils import load_data
import wandb
import torchattacks

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
   
    opt.len_data = 50000
    model = create_model(opt)      # create a model given opt.model and other options
    if opt.load_G:                 # only teacher is loaded
        model.setup(opt,only_G=True)
    else:    
        model.setup(opt)  

    
    #test_model = find_model_using_name("test")
    #test_model.setup(opt)


    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    attack=None

    if opt.reg == 'cosim_KL':            #in cosim_KL loss we adversarial image for training
        if os.path.basename(opt.dataroot) =="cifar_10":
            print('Attack using PGD on cifar10 dataset')
            eps, alpha, steps = 8/255, 2/255, 7
        elif  os.path.basename(opt.dataroot) =="mnist":
            print('Attack using PGD on mnist dataset')
            eps, alpha, steps = 0.3, 0.01, 40
        elif  os.path.basename(opt.dataroot) =="fmnist":
            print('Attack using PGD on fmnist dataset')
            #eps, alpha, steps = 0.3, 0.01, 100
            eps, alpha, steps = 0.2, 0.02, 40
        elif  os.path.basename(opt.dataroot) =="svhn":
            print('Attack using PGD on fmnist dataset')
            #eps, alpha, steps = 0.3, 0.01, 100
            eps, alpha, steps = 0.02, 0.02/10 ,10

        if opt.attack == 'PGD':        
            attack = torchattacks.PGD(model.netT, eps=eps, alpha=alpha, steps=steps)

    dataset, testset = load_data(batch_size=opt.batch_size, transform_type='Adv', translated_path='l2keep15inl2_togenl1_G60D1_keep15', transform=opt.reg_transform,
     use_synthetic_dataset=  opt.use_synthetic_dataset, synth_root = opt.synth_root , adv_attack=attack , dataroot=opt.dataroot)
        # Added
    _, testset1 = load_data(batch_size=opt.batch_size, dataroot=opt.dataroot)     # this test dataset is used for testing. it is a plain dataset without any transform
    dataset_size = len(dataset)    # get the number of images in the dataset.
    
    print('The number of training images = %d' % dataset_size)
    model.set_toy_imgs(dataset)    # Added
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.init_metrics()            # Added
        model.reinit_D_metrics()        # Added
        for i, data in enumerate(dataset):  # inner loop within one epoch
            
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data, epoch=epoch)         # unpack data from dataset and apply preprocessing
            if opt.reg=='discriminator' or opt.reg=='discriminator_cosim' or opt.reg=='discriminator_perceptual' or opt.reg=='discriminator_teacher' or opt.reg=='discriminator_spectral':
                if (total_iters+(opt.D_update_freq*opt.batch_size)-opt.batch_size)%(opt.D_update_freq*opt.batch_size)==0 :   # Added
                    model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                else :   # Added
                    # her opt_D= False means discriminator's weights are not updated only generator is trained
                    model.optimize_parameters(opt_D=False)   # Added
            else :
                # her opt_D= False means discriminator's weights are not updated only generator is trained
                model.optimize_parameters(opt_D=False) 
            
            #TODO uncomment this block later
            """if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                with torch.no_grad():   # Added
                    model.toy_fake_B = model.netG(model.toy_real_B)    # Added
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)"""

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                model.eval_teacher()  
                losses = model.get_current_losses()
                model.reinit_D_metrics()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            

            iter_data_time = time.time()
            break
        if epoch %1  == 0:              # cache our model every <save_epoch_freq> epochs
            
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
              # test model after every epoch
            model.init_metrics()            # Added
            model.reinit_D_metrics()        # Added
            for i, data in enumerate(testset1):
                
                model.set_input(data, attack = attack) # unpack data from data loader
                model.test() # run
                model.eval_teacher()
           
            metrics_s = {
                'epoch' :epoch,
                'test_clean_wv_acc' : (model.lp_metrics['correct'] / model.lp_metrics['total']) * 100,
                'test_clean_acc' : (model.og_metrics['correct'] / model.og_metrics['total']) * 100, 
                'test_clean_wv_I2I_acc' : (model.gen_metrics['correct'] / model.gen_metrics['total']) * 100,
            }
            print(metrics_s)
            wandb.log(metrics_s)
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
