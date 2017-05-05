from __future__ import print_function
import torch.utils.data as data
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import csv
import os
import glob

from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

class imageNet(data.Dataset):
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
            try:
                self.generateTorchFile()
            except:
                raise RuntimeError('Unable to use generate torch file')

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
        img = Image.fromarray(img.numpy(), mode='L')

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

    def generateTorchFile(root,raw_folder,processed_folder,training_file,test_file):
        print('Processing..')
        '''Convert dataset folder into pytorch format'''
        training_set = (
            read_image_folder(os.path.join(root, raw_folder, 'train/images/')),
            read_label_file(os.path.join(root, raw_folder, 'train/train_labels.csv'))
        )
        test_set = (
            read_image_folder(os.path.join(root, raw_folder, 'test/images/')),
            #read_label_file(os.path.join(root, raw_folder, 'test-labels'))
            torch.LongTensor(10000)
        )
        with open(os.path.join(root, processed_folder, training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(root, processed_folder, test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done')


def read_label_file(path):
    '''Load the labels into a long tensor'''
    labels = []
    reader = csv.reader(open(path))
    #Skip first line, containing text
    next(reader)    
    #sortedList = sorted(reader,key=lambda row: row[0],reverse = False)
    for name,label in reader:
        labels.append(int(label))
    return torch.LongTensor(labels)


def read_image_folder(path):
    '''Load all images in path into tensor'''
    images = []
    tensorTransform = ToTensor()
    for filename in glob.glob(path+'*.JPEG'): 
        im=Image.open(filename)
        imTensor = tensorTransform(im)
        images.append(imTensor)
    #return torch.ByteTensor(images).view(-1, 3, 56,56)
    return images
if __name__ == '__main__':
    #read_label_file('data/raw/train/train_labels.csv')
    #read_image_folder('data/raw/train/images/')
    #generateTorchFile('data','raw','processed','training.pt','test.pt')
    net = imageNet('data')
    print(net[0])
    show(utils.make_grid(net[0]))
    plt.show()