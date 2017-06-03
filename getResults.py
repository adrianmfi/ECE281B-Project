import torch
print("=> loading checkpoint")
checkpoint = torch.load('saved_models/checkpoint.pth.tar')
bestPrecision = float(checkpoint['best_precision'])
print('Best prediction: ',bestPrecision)
testAcc = checkpoint[testAcc]
testLoss = checkpoint[testLoss]
valAcc = checkpoint[valAcc]
valLoss = checkpoint[valLoss]

print(testAcc)

print("=> loaded checkpoint (epoch {})"
	  .format(checkpoint['epoch']))