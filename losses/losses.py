if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    print(sys.path)


import torch
import torch.nn as nn
import torch.nn.functional as F

from model_stealing.predictors import Alexnet


class DBMA_Loss(nn.Module):

    def __init__(self, losses="cosim_kl_wc"):
        super(DBMA_Loss, self).__init__()
        self.losses = losses.split("_")
        self.wc = torch.nn.L1Loss()
        self.cosim = lambda x, y: 1 - (F.cosine_similarity(x, y)).mean()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

        self.cosim_lambda = 1
        self.kl_lambda = 1
        self.wc_lambda = 1
        self.kl_temp = 1

    def forward(self, predictions):

        loss_dict = {}

        if "cosim" in self.losses:
            loss_dict["cosim"] = self.cosim_lambda * self.cosim(predictions["pred_clean_images"],
                                                                predictions["pred_clean_regen_images"].detach())

        if "kl" in self.losses:
            loss_dict["kl"] = self.kl_lambda * self.kl_criterion(
                F.log_softmax(predictions["pred_adv_regen_images"] / self.kl_temp, dim=1),
                F.softmax(predictions["pred_clean_regen_images"].detach() / self.kl_temp, dim=1)) * (
                                      self.kl_temp * self.kl_temp)

        if "wc" in self.losses:
            loss_dict["wc"] = self.wc_lambda * self.wc(predictions["clean_regen_images"], predictions[
                "clean_images"].detach())  # wc loss is computed on normalized images

        return sum(loss_dict.values()), loss_dict


if __name__ == "__main__":
    from dbma.WNR import WNR

    from dbma.unet import UnetGenerator, get_norm_layer
    from dbma.dbma import DBMA

    wnr = WNR()
    norm_layer = get_norm_layer("instance")
    regen = UnetGenerator(3, 3, 5, 64, norm_layer, False)
    victim_model = Alexnet()
    dbma = DBMA(wnr, regen)
    x = torch.rand(size=(32, 3, 32, 32))
    y = torch.rand(size=(32, 3, 32, 32))
    out = dbma(x)
    predictions={}
    for k, v in out.items():
        prediction = victim_model(v)
        key = "pred_clean_" + k
        predictions[key] = prediction

        key = "pred_adv_" + k
        predictions[key] = prediction

    l = DBMA_Loss()
    loss, _ = l(predictions)
    loss.backward()
