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

    def forward(self, clean_images):
        output = {}

        wv_clean_images = self.wnr(clean_images)  # output of WNR
        regen_clean_images = self.regen_network(wv_clean_images)  # output of regen network
       

        output["images"] = clean_images
        output["wv_images"] = wv_clean_images  # normalized images
        output["regen_images"] = regen_clean_images

        return output
