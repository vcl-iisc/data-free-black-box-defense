import numpy as np
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class WNR(nn.Module):

    def __init__(self, wavelet="db1", mode="symmetric", levels=2, keep_percentage=16):
        super(WNR, self).__init__()
        self.dwt = DWTForward(J=levels, mode=mode, wave=wavelet)  # Accepts all wave types available to PyWavelets
        self.iwt = DWTInverse(mode=mode, wave=wavelet)  # Accepts all wave types available to PyWavelets
        self.levels = levels
        self.keep = keep_percentage / 100

    def forward(self, images):

        approx_coeffs, detail_coeffs = self.dwt(images)  # forward wavelet transform

        batch_size = images.size(0)
        channels = images.size(1)

        temp = detail_coeffs[0].view(batch_size, channels, -1)  # reshape (h,w) -> (h*w)

        for l in range(1, self.levels):
            temp = torch.cat((temp, detail_coeffs[l].view(batch_size, channels, -1)), dim=2)

        Csort, _ = torch.sort(torch.abs(temp), dim=2)
        s = Csort.size(2)

        thresh = Csort[:, :, int(np.floor((1 - self.keep) * s))].view(batch_size, channels, 1, 1, 1)
        for l in range(0, self.levels):
            ind = torch.abs(detail_coeffs[l]) > thresh
            detail_coeffs[l] *= ind
        images = self.iwt((approx_coeffs, detail_coeffs))
        return images


if __name__ == "__main__":
    x = torch.rand(size=(32, 3, 32, 32))
    wnr = WNR()
    y = wnr(x)
    print(torch.sum(abs(y - x)))
