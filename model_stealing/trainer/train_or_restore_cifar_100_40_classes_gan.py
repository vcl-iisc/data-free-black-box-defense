import setup
import torch


def train_or_restore_cifar_100_40_classes_gan(sngan):
    sngan.load_state_dict(
        torch.load('./checkpoints/cifar_100_40_classes_gan.pth', map_location=setup.device)['gen_state_dict'])
    sngan.to(setup.device)
    sngan.eval()

    def visualize():
        import numpy as np
        import matplotlib.pyplot as plt
        while 1:
            image = sngan(
                torch.Tensor(np.random.normal(size=(1, 128))).cuda()
            )
            image = image.clamp(-1, 1) / 2. + .5
            image = image.detach().cpu().numpy().transpose([0, 2, 3, 1])[0]
            plt.imshow(image)
            plt.show()

    # visualize()

    return sngan
