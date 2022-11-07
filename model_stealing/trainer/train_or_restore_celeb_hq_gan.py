import torch


def celeb_gan(gan):
    """model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
        'PGAN',
        model_name='celebAHQ-256',
        pretrained = True,
        useGPU = True
    )"""
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=True)
    return model
