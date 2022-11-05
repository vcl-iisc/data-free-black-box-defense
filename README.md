
#  data-free-black-box-defense
Adversarial defense of black box models in data free setup [Paper](https://arxiv.org/abs/2211.01579)

## Prerequisites
- Download blackbox models, generative models , and DBMA checkpoints from https://drive.google.com/drive/u/0/folders/18WwA6bT4N4vwevRVYGodbQMezudjfjD3

- Store DBMA model checkpoints in directory ./checkpoints

- Store  black-box and generative model checkpoints in ./model_stealing/checkpoints/

- Model stealing code is adapted from (black box ripper)[https://github.com/antoniobarbalau/black-box-ripper]

## Setup
Following commands train DBMA for 

Teacher : Alexnet ,  Defender surroate :  Resnet18 
 , Attacker Surrogate : Alexnet_half
 , Dataset : SVHN


#### Train Defender Surrogate model

``` 
cd model_stealing
python base_experiment.py --true_dataset=svhn --teacher=alexnet --student=resnet18 --generator=cifar_100_90_classes_gan --gpu_id=0 --use_new_optimization --epochs 100 

```

#### Generate surrogate dataset

```
python generate_batch.py --true_dataset=svhn --teacher=alexnet --student=resnet18 --generator=cifar_100_90_classes_gan --gpu_id=0  --save_path ../data/teacher_alexnet_student_resnet18_svhn_synthetic --use_new_optimization

```

#### Find optimal value of K
```
cd..
python plot_roc.py --dataset svhn --attack PGD --wvlt db1 --mode symmetric --levels 2 --keep_percentage 1 --batch_size 2048 --gpu_id 0   --victim_model_path model_stealing/checkpoints/teacher_alexnet_for_svhn_state_dict --victim_model_name alexnet --surrogate_model_path model_stealing/checkpoints/student_resnet18_teacher_alexnet_svhn_cifar_100_90_classes_gan_adam_75_state_dict --surrogate_model_name resnet18 --synthetic_dataset_path ./data/teacher_alexnet_student_resnet18_svhn_synthetic/
```

    Plot of graph of roc values. At optimal k, ROC curve saturates. k=20


#### train regenerator network
 
```
python train.py --dataset svhn --name Alexnet_resnet18_svhn_cosim_kl_wc_2_20 --attack PGD --wvlt db1 --mode symmetric --levels 2 --keep_percentage 20 --batch_size 512 --n_epochs 140  --loss cosim_kl_wc --lr_policy linear --lr 0.0002   --n_epochs_decay 160 --gpu_id 0   --surrogate_model_path  model_stealing/checkpoints/student_resnet18_teacher_alexnet_svhn_cifar_100_90_classes_gan_adam_75_state_dict --surrogate_model_name resnet18 --synthetic_dataset_path data/teacher_alexnet_student_resnet18_svhn_synthetic/ 
```


#### Create Attacker surrogate model
```
cd model_stealing
python base_experiment.py --true_dataset=svhn --teacher=dbma --student=alexnet_half --generator=cifar_100_90_classes_gan --gpu_id=0 --use_new_optimization --epochs 75 --wvlt db1 --mode symmetric --levels 2 --keep_percentage 20 --victim_model_path checkpoints/teacher_alexnet_for_svhn_state_dict --victim_model_name alexnet --dbma_path  ../checkpoints/Alexnet_resnet18_svhn_cosim_kl_wc_2_20/15.pth --dbma_name Alexnet_resnet18_svhn_cosim_kl_wc_2_20
```

#### Test 
```
cd ..
python test.py --dataset svhn --attack PGD --wvlt db1 --mode symmetric --levels 2 --keep_percentage 20 --batch_size 512 --epoch 15 --gpu_id 0 --name Alexnet_resnet18_svhn_cosim_kl_wc_2_20  --victim_model_path model_stealing/checkpoints/teacher_alexnet_for_svhn_state_dict --victim_model_name alexnet --surrogate_model_path model_stealing/checkpoints/student_alexnet_half_teacher_dbma_svhn_cifar_100_90_classes_gan_Alexnet_resnet18_svhn_cosim_kl_wc_2_20_adam_100_state_dict --surrogate_model_name alexnet_half

```

### Test with Pretrained models
Download required checkpoints of dbma and surrogate model from
[drive](https://drive.google.com/drive/u/0/folders/18WwA6bT4N4vwevRVYGodbQMezudjfjD3)

run test script by replacing the paths 






## Acknowledgements

 - [Black box model stealing](https://github.com/antoniobarbalau/black-box-ripper)
 - [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/readme.html)
 - [Towards Data-Free Model Stealing in a Hard Label Setting](https://github.com/val-iisc/Hard-Label-Model-Stealing)


## Authors

- [Shubham Randive](https://github.com/shubham303)
- [Inder Khatri]
- [Gaurav Nayak]
- 