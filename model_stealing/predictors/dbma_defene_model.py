"""
New Model created with combination of DBMA and vicitm model
"""

import torch.nn as nn

class DBMADefense(nn.Module):

    def __init__(self,dbma, victim_model):
        super(DBMADefense, self).__init__()
        self.dbma  = dbma
        self.victim_model = victim_model
        self.name = "dbma"
        self.n_outputs=10

    def forward(self, images):
        output = self.dbma(images)
        return self.victim_model(output["regen_images"])



