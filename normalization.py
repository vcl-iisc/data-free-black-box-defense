import torch
import torch.nn as nn


class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super(Denormalize, self).__init__()
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(1, len(mean), 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(1, len(mean), 1, 1), requires_grad=False)

    def forward(self, input):
        return (input * self.std) + self.mean


if __name__ == "__main__":
    n = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(torch.device(0))
    x = torch.randn((32, 3, 32, 32)).to(torch.device(0))
    y = n(x)
