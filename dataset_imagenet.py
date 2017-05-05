from __future__ import print_function
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import csv
import os
import glob
import operator


class ImageNet(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if self.train:
            self.labelcsv = csv.reader(open(os.path.join(self.root, self.raw_folder, 'train/train_labels.csv')))
            self.labelcsv = sorted(self.labelcsv,key=operator.itemgetter(0))
            self.imgurls = sorted(glob.glob(os.path.join(self.root, self.raw_folder, 'train/images/')+'*.JPEG'))
        else:
            #self.labelcsv = csv.reader(open(os.path.join(self.root, self.raw_folder, 'test/train_labels.csv')))
            self.imgurls = sorted(glob.glob(os.path.join(self.root, self.raw_folder, 'test/images/')+'*.JPEG'));
    def __getitem__(self, index):
        img = Image.open(self.imgurls[index]).convert('RGB')
        if self.train:
            target = torch.LongTensor([int(self.labelcsv[index][1])])
        else:
            target = torch.LongTensor([-1])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return 50000
        else:
            return 10000

if __name__ == '__main__':
    #read_label_file('data/raw/train/train_labels.csv')
    #read_image_folder('data/raw/train/images/')
    #generateTorchFile('data','raw','processed','training.pt','test.pt')
    net = ImageNet('data',)
    net[49999][0].show()