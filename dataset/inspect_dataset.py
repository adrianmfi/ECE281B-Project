import torch
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
