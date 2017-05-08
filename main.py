from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import os
import numpy as np
from dataset_imagenet import ImageNet
from torchvision import utils

def train(epoch,model,optimizer,trainLoader):
    model.train()
    for batchIdx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)    
        optimizer.zero_grad() 
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batchIdx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.data[0]))

def validate(epoch,model,optimizer,valLoader):
    model.eval()
    valLoss = 0
    correct = 0
    for idx,(data, target) in enumerate(valLoader):##
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valLoss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    valLoss = valLoss
    valLoss /= len(valLoader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valLoss, correct, len(valLoader.dataset),
        100. * correct / len(valLoader.dataset)))

    return correct/len(valLoader.dataset)

def saveCheckpoint(state, isBest, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if isBest:
        shutil.copyfile(filename, 'models/model_best.pth.tar')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='ECE281 CNN Image classification')
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

    #Use CUDA if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    #Set up the data loaders, optimizer and net
    trainLoader = torch.utils.data.DataLoader(
        ImageNet('data', train=True,transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])), 
        batch_size=args.batch_size, shuffle=True, **kwargs)
    valLoader = torch.utils.data.DataLoader(
        ImageNet('data', train=False, transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 100)

    optimizer = optim.SGD(model_conv.fc.parameters(), lr=args.lr, momentum=args.momentum)
    startEpoch = 1
    bestPrecision = 0
    #Resume from checkpoint if possible
    if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                startEpoch = checkpoint['epoch']
                #bestPrecision = checkpoint['best_precision']
                model.load_state_dict(checkpoint['state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint (epoch {})"
                      .format(checkpoint['epoch']))
    if args.cuda:
        model.cuda()
             
    for epoch in range(startEpoch, startEpoch+args.epochs + 1):
        train(epoch,model_conv,optimizer,trainLoader)
        precision = validate(epoch,model_conv,optimizer,valLoader)
        isBest = precision > bestPrecision
        saveCheckpoint({
            'epoch': epoch + 1,
            'state_dict': model_conv.state_dict(),
            'best_precision': bestPrecision,
            'optimizer' : optimizer.state_dict(),
        }, isBest)
