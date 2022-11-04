if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    print(sys.path)

import torch.nn as nn


class DBMA(nn.Module):

    def __init__(self, wnr, regenerator_network):
        super(DBMA, self).__init__()
        self.wnr = wnr
        self.regen_network = regenerator_network

    def forward(self, clean_images, adv_images=None):
        output = {}

        wv_clean_images = self.wnr(clean_images)  # output of WNR
        regen_clean_images = self.regen_network(wv_clean_images)  # output of regen network

        output["clean_images"] = clean_images
        output["wv_clean_images"] = wv_clean_images  # normalized images
        output["regen_clean_images"] = regen_clean_images

        if adv_images is not None:
            # adv_images = self.norm(adv_images)
            wv_adv_images = self.wnr(adv_images)  # output of WNR
            regen_adv_images = self.regen_network(wv_adv_images)  # output of regen network

            output["adv_images"] = adv_images
            output["regen_adv_images"] = regen_adv_images
            output["wv_adv_images"] = wv_adv_images

        return output
