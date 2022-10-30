import torchattacks
import torch
import torchvision

def cat(tensor1, tensor2, dim=0):
    if tensor1 is None:
        return tensor2
    else:
        return torch.cat((tensor1, tensor2), dim=dim)



def main(loader):
    x =None
    y=None

    for i, (imgs, labels) in enumerate(loader):
        
        x = cat(x, imgs)
        y = cat(y, labels)  
    
    torch.save(x,"/media2/inder/GenericRobustness/cifar_10_real_dataset_converted/images.pt")
    torch.save(y, "/media2/inder/GenericRobustness/cifar_10_real_dataset_converted/labels.pt")

if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform = torchvision.transforms.ToTensor())
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=False, num_workers=4, drop_last=True)
    #return adversarial image to dataset output
    main(trainloader)