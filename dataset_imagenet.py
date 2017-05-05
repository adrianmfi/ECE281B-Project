from __future__ import print_function
import torch.utils.data as data
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import csv
import os
import glob
import operator

import matplotlib.pyplot as plt

class ImageNet(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_exists():
            self.generateTorchFile()

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        tf = ToPILImage()
        img = tf(img)
        
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

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def generateTorchFile(self):
        print('Processing..')
        '''Convert dataset/raw folder into pytorch format and save into dataset/processed as .pt files'''
        training_set = (
            read_image_folder(os.path.join(self.root, self.raw_folder, 'train/images/')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train/train_labels.csv'))
        )
        test_set = (
            read_image_folder(os.path.join(self.root, self.raw_folder, 'test/images/')),
            #read_label_file(os.path.join(root, raw_folder, 'test-labels'))
            torch.LongTensor(10000)
        )
        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            os.makedirs(os.path.join(self.root, self.processed_folder))
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done')

# Should probabaly make sure labels and pictures match..
def read_label_file(path):
    '''Load the labels into a long tensor'''
    labels = []
    reader = csv.reader(open(path))
    #Skip first line, containing text
    next(reader)    
    #sortedList = sorted(reader,key=lambda row: row[0],reverse = False)
    for name,label in sorted(reader,key=operator.itemgetter(0)):
        labels.append(int(label))
    return torch.LongTensor(labels)

def read_image_folder(path):
    '''Load all images in path folder into tensor'''
    images = []
    for filename in sorted(glob.glob(path+'*.JPEG')): 
        im=Image.open(filename)
        im = np.array(im.getdata(),dtype = np.uint8).reshape(-1,56,56)
        #Some of the images only have 1 channel?
        if (im.shape[0] == 3):
            images.append(im)
        else:
            imcolor= np.concatenate((im,im,im),axis = 0)
            images.append(imcolor)
    return torch.ByteTensor(np.array(images)) #Convert to HWC
    #return images

if __name__ == '__main__':
    #read_label_file('data/raw/train/train_labels.csv')
    #read_image_folder('data/raw/train/images/')
    #generateTorchFile('data','raw','processed','training.pt','test.pt')
    net = ImageNet('data')
    imgs,labels = net[0:3]
    print( imgs)
