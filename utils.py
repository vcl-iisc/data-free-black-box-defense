import torch

from model_stealing.predictors import Alexnet, AlexnetHalf
from model_stealing.predictors.dbma.dbma import DBMA
from model_stealing.predictors.dbma.WNR import WNR
from model_stealing.predictors.dbma.unet import UnetGenerator, get_norm_layer
from model_stealing.predictors import ResNet18

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

def create_dbma_model(args=None):
    wnr = WNR(args.wvlt, args.mode, args.levels, args.keep_percentage)
    norm_layer = get_norm_layer("instance")
    regen = UnetGenerator(3, 3, 5, 64, norm_layer, False)

    if args.victim_model_name is not None:
        victim_model = get_model(args.victim_model_name, args.victim_model_path)
    else:
        victim_model = None

    surrogate_model = get_model(args.surrogate_model_name, args.surrogate_model_path)
    dbma = DBMA(wnr, regen, surrogate_model, victim_model)
    return dbma


if __name__ == "__main__":

    wnr = WNR()
    norm_layer = get_norm_layer("instance")
    regen = UnetGenerator(3, 3, 5, 64, norm_layer, False)
    model = ResNet18()
    dbma = DBMA(wnr, regen, model)
    x = torch.rand(size=(32, 3, 32, 32))
    y = torch.rand(size=(32, 3, 32, 32))
    out = dbma(x, y)
    print(out.keys())
