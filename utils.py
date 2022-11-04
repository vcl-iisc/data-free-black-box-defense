import torch

from model_stealing.predictors import Alexnet, AlexnetHalf, ResNet18


def get_model(name, path):
    # return model with normalized layer on top of it.
    if name == "alexnet":
        model = Alexnet()
    elif name == "alexnet_half":
        model = AlexnetHalf()
    elif name == "resnet18":
        model = ResNet18()
    else:
        raise Exception("unknown models {}".format(name))

    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
