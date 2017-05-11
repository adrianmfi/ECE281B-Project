from __future__ import print_function
import torch.utils.data as data
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import csv
import os
import glob
import operator
import shutil

cutoff = 40000

class ImageNet(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or validation set

        if self.train:
            self.data, self.labels = torch.load(root+'/processed/train.pt')
        else:
            self.data, self.labels = torch.load(root+'/processed/validate.pt')

    def __getitem__(self, index):
        img,target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img.numpy(),(1,2,0)))
        print(img)
        img.show()
        exit(1)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return cutoff
        else:
            return 50000-cutoff

def preprocessData():
    #Sorts and splits the training set and its corresponding labels
    #into two folders with cutoff number of training pictures in the first
    #and 50000 - cutoff numbers in the other
    #Requires the csv file to be stored in ./data/raw/train
    #and the images in ./data/raw/train/images
    #Also resizes the image if specified
    print('Processing...')
    #If images should be resized
    resize = True
    newSize = 224;

    labelcsv = csv.reader(open(os.path.join('data', 'raw', 'train/train_labels.csv')))
    #Skip header line
    next(labelcsv)

    imgSize = newSize if resize else 56
    images_train = torch.ByteTensor(cutoff,3,imgSize,imgSize)
    labels_train = torch.zeros(cutoff)
    images_val = torch.ByteTensor(50000-cutoff,3,imgSize,imgSize)
    labels_val = torch.zeros(50000-cutoff)
    toTensor = ToTensor()
    
    i = 0
    for entry in labelcsv:
        if i >= cutoff:
            break
        filename = entry[0]
        label = int(entry[1])
        img = Image.open(os.path.join('data/raw/train/images',filename+'.JPEG')).convert("RGB")
        if resize:
            img = img.resize((newSize,newSize),Image.BICUBIC)
        imgTensor = toTensor(img)
        imgBytes = imgTensor.byte()
        images_train[i] = imgBytes
        labels_train[i] = label
        i+=1
    i = 0
    for entry in labelcsv:
        if i >= 50000-cutoff:
            break
        filename = entry[0]
        label = int(entry[1])
        img = Image.open(os.path.join('data/raw/train/images',filename+'.JPEG')).convert("RGB")
        if resize:
            img = img.resize((newSize,newSize))
        imgTensor = toTensor(img)
        imgBytes = imgTensor.byte()
        images_val[i] = imgBytes
        labels_val[i] = label
        i+=1
    if not os.path.exists('data/processed/'):
        os.makedirs('data/processed/')
    with open('data/processed/train.pt', 'wb') as f:
        torch.save((images_train,labels_train), f)
    with open('data/processed/validate.pt', 'wb') as f:
        torch.save((images_val,labels_val), f)
    print('Done!')
if __name__ == '__main__':
    preprocessData()