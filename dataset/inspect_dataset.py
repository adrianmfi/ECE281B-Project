import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import PIL
import operator

def inspectProcessed():
	#Counts the pictures of each class and shows pictures from class imageNum
	calcHist = False
	imageNum = 1


	data_train, labels_train = torch.load('data/processed/train.pt')
	data_val, labels_val = torch.load('data/processed/validate.pt')

	if calcHist:
		hist_train = np.zeros(100)
		hist_val = np.zeros(100)

		for label in labels_train:
			index = int(label)
			hist_train[index] = hist_train[index] +1

		for label in labels_val:
			index = int(label)
			hist_val[index] = hist_val[index] +1

		print(hist_train)
		print(hist_val)

	for d,l in zip(data_train,labels_train):
		if int(l) == imageNum:
			img = d.numpy()
			plt.imshow(img)
			plt.show()

def inspectRaw():
	#Counts how many bw pictures are of each class
	labelcsv = csv.reader(open('../data/raw/train/train_labels.csv'))
	next(labelcsv)
	bwcounts = np.zeros(100)
	labels = np.zeros(100)
	for i,row in enumerate(sorted(labelcsv,key = operator.itemgetter(0))):
		path = row[0]
		label = int(row[1])
		img = PIL.Image.open('../data/raw/train/images/'+path+'.JPEG')
		if img.mode != 'RGB':
			bwcounts[label] +=1
		labels[label] +=1
		if i == 1000:
			break
	print(bwcounts)
	print(labels)

def testUnshuffled():
if __name__ == '__main__':
	inspectRaw()