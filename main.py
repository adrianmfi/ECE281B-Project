from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import math
import time
import shutil
import os
import numpy as np
from dataset_imagenet import ImageNet
from torchvision import utils

# Training settings
parser = argparse.ArgumentParser(description='ECE281 CNN Image classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--workers', type=int, default=0, metavar='N',
                    help='Number of workers (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()


#Use CUDA if available
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Set up the data loaders, optimizer and net
trainLoader = torch.utils.data.DataLoader(
    ImageNet('data', train=True,transform = transforms.Compose([
        transforms.Scale(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.478571, 0.44496, 0.392131],[0.26412, 0.255156, 0.269064])
        ])), 
    batch_size=args.batch_size, num_workers = args.workers, shuffle=True, **kwargs)
valLoader = torch.utils.data.DataLoader(
    ImageNet('data', train=False, transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize([0.478571, 0.44496, 0.392131],[0.26412, 0.255156, 0.269064])
        ])),
    batch_size=args.batch_size, num_workers = args.workers, shuffle=True, **kwargs)

#Set up the model and optimizer
model_conv = WideResNet(22, 100,4)
print(model_conv)
optimizer = optim.SGD(model_conv.fc.parameters(), lr=args.lr, momentum=args.momentum)
startEpoch = 1
bestPrecision = 0

#Resume from checkpoint if specified
if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            startEpoch = checkpoint['epoch']
            bestPrecision = float(checkpoint['best_precision'])
            print('Best prediction: ',bestPrecision)
            model_conv.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
if args.cuda:
    model_conv.cuda()
def train(epoch):
    exp_lr_scheduler(epoch, args.lr)
    model_conv.train()
    for batchIdx, (data, target) in enumerate(trainLoader):
        #Because of bug in dataset, preprocess again and remove
        target = target.long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)    
        optimizer.zero_grad() 
        output = model_conv(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batchIdx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.data[0]))

def validate(epoch):
    model_conv.eval()
    valLoss = 0
    correct = 0
    for idx,(data, target) in enumerate(valLoader):##
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.long()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model_conv(data)
        valLoss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    valLoss = valLoss
    valLoss /= len(valLoader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valLoss, correct, len(valLoader.dataset),
        100. * correct / len(valLoader.dataset)))

    return correct/len(valLoader.dataset)

def exp_lr_scheduler(epoch, init_lr, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def saveCheckpoint(state, isBest, filename='saved_models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if isBest:
        shutil.copy(filename, 'saved_models/model_best.pth.tar')


#Train
for epoch in range(startEpoch, startEpoch+args.epochs + 1):
    startTime = time.clock()
    train(epoch)
    endTime = time.clock()
    print ('Time used per epoch: ',(endTime-startTime))
    precision = validate(epoch)
    isBest = False
    if precision > bestPrecision:
        bestPrecision = precision
        isBest = True 
    saveCheckpoint({
        'epoch': epoch + 1,
        'state_dict': model_conv.state_dict(),
        'best_precision': bestPrecision,
        'optimizer' : optimizer.state_dict(),
    }, isBest)


