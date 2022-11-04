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
            loss_dict["cosim"] = self.cosim_lambda * self.cosim(predictions["pred_clean"],
                                                                predictions["pred_regen_clean"])

        if "kl" in self.losses:
            loss_dict["kl"] = self.kl_lambda * self.kl_criterion(
                F.log_softmax(predictions["pred_regen_adv"] / self.kl_temp, dim=1),
                F.softmax(predictions["pred_regen_clean"].detach() / self.kl_temp, dim=1)) * (
                                      self.kl_temp * self.kl_temp)

        if "wc" in self.losses:
            loss_dict["wc"] = self.wc_lambda * self.wc(predictions["regen_clean_images"], predictions[
                "clean_images"])  # wc loss is computed on normalized images

        return sum(loss_dict.values()), loss_dict


if __name__ == "__main__":
    from model_stealing.predictors.dbma.WNR import WNR

    from model_stealing.predictors.dbma.unet import UnetGenerator, get_norm_layer
    from model_stealing.predictors.dbma.dbma import DBMA

    wnr = WNR()
    norm_layer = get_norm_layer("instance")
    regen = UnetGenerator(3, 3, 5, 64, norm_layer, False)
    victim_model = Alexnet()
    dbma = DBMA(wnr, regen, victim_model)
    x = torch.rand(size=(32, 3, 32, 32))
    y = torch.rand(size=(32, 3, 32, 32))
    out = dbma(x, y)
    l = DBMA_Loss()
    loss, _ = l(out)
    loss.backward()
