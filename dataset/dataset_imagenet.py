from __future__ import print_function
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import csv
import os
import glob
import operator 
cutoff = 40000

class ImageNet(data.Dataset):

    raw_folder = 'raw'
    processed_folder = 'processed'
    def __init__(self, root, mode='train', transform=None, target_transform=None,fromFolder = True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode  # training, test or validation set
        self.fromFolder = fromFolder
        self.trainLen = cutoff
        self.valLen = 50000-cutoff
        if self.fromFolder:
            if self.mode == 'train':
                self.labelcsv = csv.reader(open(os.path.join(self.root, self.processed_folder, 'train/labels.csv')))
                self.labelcsv = sorted(self.labelcsv,key=operator.itemgetter(0))
            elif self.mode == 'validate':
                self.labelcsv = csv.reader(open(os.path.join(self.root, self.processed_folder, 'validate/labels.csv')))
                self.labelcsv = sorted(self.labelcsv,key=operator.itemgetter(0))
            elif self.mode == 'test':
                self.imgurls = glob.glob(os.path.join(self.root,self.processed_folder, 'test/images/')+'*.JPEG')
            else:
                print('unknown mode')
        else:    
            if self.mode == 'train':
                self.data, self.labels = torch.load(root+'/processed/train.pt')
            elif self.mode == 'validate':
                self.data, self.labels = torch.load(root+'/processed/validate.pt')
            elif self.mode == 'test':
                pass
            else:
                print ('unknown mode')
    def __getitem__(self, index):
        if self.fromFolder:
            if self.mode == 'train':
                url = self.labelcsv[index][0]
                target = int(self.labelcsv[index][1])
                img = Image.open(self.root+'/'+self.processed_folder+'/train/images/'+url+'.JPEG')
            elif self.mode == 'validate':
                url = self.labelcsv[index][0]
                target = int(self.labelcsv[index][1])
                img = Image.open(self.root+'/'+self.processed_folder+'/validate/images/'+url+'.JPEG')
            else:
                target = self.imgurls[index].lstrip(os.path.join(self.root,self.processed_folder, 'test/images/')).rstrip('.JPEG')
                img = Image.open(self.imgurls[index])
        else:        
            img,target = self.data[index], self.labels[index]
            img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return self.trainLen
        elif self.mode == 'validate':
            return self.valLen
        else:
            return 10000

def preprocessData():
    #Splits the training set and its corresponding labels
    #into two folders with cutoff number of training pictures in the first
    #and 50000 - cutoff numbers in the other
    #Requires the csv file to be stored in ./data/raw/train
    #and the images in ./data/raw/train/images
    #Also resizes the image if specified
    print('Processing...')
    #If images should be resized
    resize = True
    newSize = 224
    toFolder = True

    labelcsv = csv.reader(open(os.path.join('data', 'raw', 'train/train_labels.csv')))
    #Skip header line
    next(labelcsv)
    labelcsv = list(labelcsv)

    imgSize = newSize if resize else 56
    images_train = torch.ByteTensor(cutoff,imgSize,imgSize,3)
    labels_train = torch.LongTensor(cutoff)
    images_val = torch.ByteTensor(50000-cutoff,imgSize,imgSize,3)
    labels_val = torch.LongTensor(50000-cutoff)
    

    if not os.path.exists('data/processed/'):
        os.makedirs('data/processed/')
    if toFolder:
        if not os.path.exists('data/processed/train'):
            os.makedirs('data/processed/train/images')
        if not os.path.exists('data/processed/validate'):
            os.makedirs('data/processed/validate/images')
        if not os.path.exists('data/processed/test'):
            os.makedirs('data/processed/test/images')

    i = 0
    for entry in labelcsv[:cutoff]:
        filename = entry[0]
        label = int(entry[1])
        img = Image.open(os.path.join('data/raw/train/images',filename+'.JPEG')).convert("RGB")
        if resize:
            img = img.resize((newSize,newSize),Image.BICUBIC)
        if toFolder:
            img.save('data/processed/train/images/'+filename+'.JPEG')        
        else:
            imgnp = np.array(img.getdata(), dtype=np.uint8).reshape(imgSize, imgSize,3)
            imgBytes = torch.ByteTensor(imgnp)        
            images_train[i] = imgBytes
            labels_train[i] = label
        i+=1
    print(i)
    i = 0
    for entry in labelcsv[cutoff:]:
        filename = entry[0]
        label = int(entry[1])
        img = Image.open(os.path.join('data/raw/train/images',filename+'.JPEG')).convert("RGB")
        if resize:
            img = img.resize((newSize,newSize))
        if toFolder:
            img.save('data/processed/validate/images/'+filename+'.JPEG')
        else: 
            imgnp = np.array(img.getdata(), dtype=np.uint8).reshape(imgSize, imgSize,3)
            imgBytes = torch.ByteTensor(imgnp)
            images_val[i] = imgBytes
            labels_val[i] = label
        i+=1
    print(i)
    if toFolder:
        for url in glob.glob('data/raw/test/images/*.JPEG'):
            img = Image.open(url).convert("RGB")
            img = img.resize((newSize,newSize))
            img.save('data/processed/test/images')
        with open('data/processed/train/labels.csv','w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(labelcsv[0:cutoff])
        with open('data/processed/validate/labels.csv','w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(labelcsv[cutoff:50000])
    else:
        with open('data/processed/train.pt', 'wb') as f:
            torch.save((images_train,labels_train), f)
        with open('data/processed/validate.pt', 'wb') as f:
            torch.save((images_val,labels_val), f)
    print('Done!')
if __name__ == '__main__':
    if not os.path.exists('data/processed/test'):
        os.makedirs('data/processed/test/images')
    for url in glob.glob('data/raw/test/images/*.JPEG'):
        img = Image.open(url).convert("RGB")
        img = img.resize((224,224))
        img.save('data/processed/test/images/'+url.lstrip('data/raw/test/images'))
    #preprocessData()
    #n = ImageNet('data')
    #i1 = n[0]
    #i2 = n[1]
    #print(i1,i2)