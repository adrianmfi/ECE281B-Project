from __future__ import print_function
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import csv
import os
import glob
import operator
import shutil

class ImageNet(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if self.train:
            self.labelcsv = csv.reader(open(os.path.join(self.root, self.processed_folder, 'train/labels.csv')))
            self.imgurls = sorted(glob.glob(os.path.join(self.root, self.processed_folder, 'train/images/')+'*.JPEG'))
        else:
            self.labelcsv = csv.reader(open(os.path.join(self.root, self.processed_folder, 'validate/labels.csv')))
            self.imgurls = sorted(glob.glob(os.path.join(self.root, self.processed_folder, 'validate/images/')+'*.JPEG'));
        self.labelcsv = sorted(self.labelcsv,key=operator.itemgetter(0))
    def __getitem__(self, index):
        #Some images are bw
        img = Image.open(self.imgurls[index]).convert('RGB')
        target = int(self.labelcsv[index][1])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return 40000
        else:
            return 10000

def splitTrainingSetInTwo():
    #Sorts and splits the training set and its corresponding labels into two folders with cutoff number of training pictures
    cutoff = 40000

    if not os.path.exists('data/processed/train/images'):
        os.makedirs('data/processed/train/images')
    if not os.path.exists('data/processed/validate/images'):
        os.makedirs('data/processed/validate/images')

    imgurls = sorted(glob.glob(os.path.join('data','raw', 'train/images/')+'*.JPEG'))
    labelcsv = csv.reader(open(os.path.join('data', 'raw', 'train/train_labels.csv')))
    labelcsv = sorted(labelcsv,key=operator.itemgetter(0))

    with open('data/processed/train/labels.csv','w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labelcsv[0:cutoff])
    with open('data/processed/validate/labels.csv','w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labelcsv[cutoff:50000])

    for url in imgurls[0:cutoff]:
        shutil.copy(url,'data/processed/train/images/')
    for url in imgurls[cutoff:50000]:
        shutil.copy(url,'data/processed/validate/images/')
    
if __name__ == '__main__':
    #net = ImageNet('data',)
    #net[49999][0].show()
    splitTrainingSetInTwo()