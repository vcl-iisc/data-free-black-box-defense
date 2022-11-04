if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    print(sys.path)

import torch
import torch.nn as nn

from model_stealing.predictors import ResNet18
from model_stealing.predictors.dbma.WNR import WNR
from model_stealing.predictors.dbma.unet import UnetGenerator, get_norm_layer
from utils import get_model


class DBMA(nn.Module):

    def __init__(self, wnr, regenerator_network, surrogate_model=None, victim_model=None):
        super(DBMA, self).__init__()
        self.wnr = wnr
        self.regen_network = regenerator_network

        self.surrogate_model = surrogate_model
        self.victim_model = victim_model

        self.surrogate_model.eval()  # testing and training surrogate model
        if self.victim_model is not None:
            self.victim_model.eval()  # victim model is not used during training

    def state_dict(self, **kwargs):
        return self.regen_network.state_dict()

    def load_state_dict(self, state_dict, **kwargs):
        self.regen_network.load_state_dict(state_dict)

    def eval(self):
        self.regen_network.eval()

    def train(self, **kwargs):
        # surrogate and victim model  is always set to eval mode
        self.regen_network.train()

    def forward(self, clean_images, adv_images=None, train=True):
        output = {}
        # during training victim model is not used.
        # During testing surrogate model is used to create adversarial samples and accuracy is computed in victim model

        if train:
            model = self.surrogate_model
        else:
            model = self.victim_model

        # clean_images = self.norm(clean_images)
        wv_clean_images = self.wnr(clean_images)  # output of WNR
        regen_clean_images = self.regen_network(wv_clean_images)  # output of regen network

        output["clean_images"] = clean_images  # normalized images
        output["regen_clean_images"] = regen_clean_images

        pred_clean = model(clean_images)
        pred_wv_clean = model(wv_clean_images)
        pred_regen_clean = model(regen_clean_images)

        output["pred_clean"] = pred_clean
        output["pred_wv_clean"] = pred_wv_clean
        output["pred_regen_clean"] = pred_regen_clean

        if adv_images is not None:
            # adv_images = self.norm(adv_images)
            wv_adv_images = self.wnr(adv_images)  # output of WNR
            regen_adv_images = self.regen_network(wv_adv_images)  # output of regen network

            output["regen_adv_images"] = regen_adv_images

            pred_adv = model(adv_images)
            pred_wv_adv = model(wv_adv_images)
            pred_regen_adv = model(regen_adv_images)

            output["pred_adv"] = pred_adv
            output["pred_wv_adv"] = pred_wv_adv
            output["pred_regen_adv"] = pred_regen_adv

        return output


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
