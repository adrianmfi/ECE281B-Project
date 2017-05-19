from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import time
import shutil
import os
import numpy as np
from dataset.dataset_imagenet import ImageNet
from torchvision import utils

# Training settings
parser = argparse.ArgumentParser(description='ECE281 CNN Image classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('--workers', type=int, default=4, metavar='N',
					help='Number of workers if CUDA is used (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--modepath', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')

def main():
	global args
	args = parser.parse_args()

	#Use CUDA if available
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	print('Cuda used: ', args.cuda)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	#Set up the data loaders
	kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
	trainLoader = torch.utils.data.DataLoader(
		ImageNet('data', train=True,transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.478571, 0.44496, 0.392131],[0.26412, 0.255156, 0.269064])
			])), 
		batch_size=args.batch_size, shuffle=True, **kwargs)

	#Set up the model, optimizer and loss function
	model = models.resnet18(pretrained= True)
	for param in model.parameters():
		print(param)
    	param.requires_grad = False
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs,100)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	if args.cuda:
		model.cuda()
		criterion = criterion.cuda()
	startEpoch = 1

	#Resume from checkpoint if specified
	if args.resume:
			if os.path.isfile(args.resume):
				print("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(args.resume)
				startEpoch = checkpoint['epoch']
				bestPrecision = float(checkpoint['best_precision'])
				print('Best prediction: ',bestPrecision)
				model.load_state_dict(checkpoint['state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer'])
				print("=> loaded checkpoint (epoch {})"
					  .format(checkpoint['epoch']))
	#Train
	for epoch in range(startEpoch, startEpoch+args.epochs + 1):
		exp_lr_scheduler(optimizer,epoch,args.lr)
		startTime = time.clock()
		train(trainLoader,model,criterion,optimizer,epoch)
		endTime = time.clock()
		print ('Time used training for epoch: ',(endTime-startTime))
		precision = validate(valLoader,model,criterion)
		isBest = False
		if precision > bestPrecision:
			bestPrecision = precision
			isBest = True
		print('Precision:', precision)
		print('Best precision:', bestPrecision)
		saveCheckpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_precision': bestPrecision,
			'optimizer' : optimizer.state_dict(),
		}, isBest)
		print()


def test(testLoader,model,criterion):
	model.eval()
	valLoss = 0
	correct = 0
	for idx,(data, target) in enumerate(valLoader):##
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		target = target.long()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		valLoss += criterion(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	valLoss = valLoss
	valLoss /= len(valLoader) # loss function already averages over batch size
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		valLoss, correct, len(valLoader.dataset),
		100. * correct / len(valLoader.dataset)))
	return correct/len(valLoader.dataset)


if __name__ == '__main__':
	main()
