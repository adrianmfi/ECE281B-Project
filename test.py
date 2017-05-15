import torch
import torchvision.datasets as dset

dset = dset.CIFAR100(root, train=True, transform=None, target_transform=None, download=True)


print(dset)