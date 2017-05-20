from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.autograd import Variable

import time
import os
import numpy as np
from dataset.dataset_imagenet import ImageNet
from torchvision import utils
import csv
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
parser.add_argument('--path', default='saved_models/model_best.pth.tar', type=str, metavar='PATH',
					help='path to latest checkpoint (default: saved_models/model_best.pth.tar)')

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
	testLoader = torch.utils.data.DataLoader(
		ImageNet('data', mode='test',transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.478571, 0.44496, 0.392131],[0.26412, 0.255156, 0.269064])
			])), 
		batch_size=args.batch_size, shuffle=False, **kwargs)

	#Set up the model
	model = models.resnet18(pretrained= True)
	for param in model.parameters():
		param.requires_grad = False
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs,100)

	#Load the model
	if os.path.isfile(args.path):
		print("=> loading checkpoint '{}'".format(args.path))
		checkpoint = torch.load(args.path)
		model.load_state_dict(checkpoint['state_dict'])
		print("=> loaded checkpoint (epoch {})"
			  .format(checkpoint['epoch']))
	else:
		print('Unable to load model')
	if args.cuda:
		model.cuda()
	#Test
	startTime = time.clock()
	precision = test(testLoader,model)
	endTime = time.clock()
	print ('Test time: ',(endTime-startTime))


def test(testLoader,model):
	model.eval()
	valLoss = 0
	correct = 0
	file = open('submission.csv','w+')
	writer = csv.writer(file)
	file = open('submission.csv','w+')
	writer = csv.writer(file)
	row = ['' for i in range(101)]
	row[0] = 'id'
	for i in range(0,100):
		row[i+1] = 'class_{num:03d}'.format(num=i)
	writer.writerow(row)
	
	for idx,(data, target) in enumerate(testLoader):##
		if args.cuda:
			data= data.cuda()
		data= Variable(data, volatile=True)
		output = model(data)
		output = F.softmax(output)

		for pred,label in zip(output,target):
			row = ['' for i in range(101)]
			row[0] = label
			row[1:] =list(pred.cpu().data)
			writer.writerow(row) 
	file.close()

if __name__ == '__main__':
	main()

