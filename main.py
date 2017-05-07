from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt 
import PIL
import numpy as np
from dataset_imagenet import ImageNet

from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

showSample = False

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    ImageNet('data', train=True,transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47858,0.44496,0.39216),(1,1,1))
        ])), 
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    ImageNet('data', train=False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47858,0.44496,0.39216),(1,1,1))
        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*3*3, 64*3*3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64*3*3, 64*3*3),
            nn.ReLU(inplace=True),
            nn.Linear(64*3*3, 100),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*3*3)
        x = self.classifier(x)
        return F.log_softmax(x)

model = Net()
if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)    
        optimizer.zero_grad() 
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for idx,(data, target) in enumerate(test_loader):##
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        # Show example of the dataset
        if idx == 0 and showSample:
            show(data,'Predicted: {}, Target: {}'.format(int(pred[0].numpy()),int(target[0].data.numpy())))##        

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def show(img,title=''):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.title(title)
    plt.show()

def findDatasetMean():
    train_loader2 = torch.utils.data.DataLoader(
        ImageNet('data', train=True,transform =transforms.ToTensor()), 
        batch_size=1)
    test_loader2 = torch.utils.data.DataLoader(
        ImageNet('data', train=False, transform =transforms.ToTensor()),
            batch_size=1)
    mean = [0,0,0]
    for data,label in train_loader2:
        mean[0] += data[0,0].mean()
        mean[1] += data[0,1].mean()
        mean[2] += data[0,2].mean()
    for data,label in test_loader2:
        mean[0] += data[0,0].mean()
        mean[1] += data[0,1].mean()
        mean[2] += data[0,2].mean()
    print (mean[0]/50000,mean[1]/50000,mean[2]/50000)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    torch.save({
            'epoch': epoch + 1,
            'arch': 'fioNet',
            'state_dict': model.state_dict(),
        }, 'checkpoint.tar' )
