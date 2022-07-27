import torch

#dataset hyperparam
imageWH = 32
channel=3
imageBatch = 16
shuffle = True
num_workers = 4

#vit hyperparam
patchWH=8
splitRow=imageWH//8
splitCol=imageWH//8
patchTotal=(imageWH//patchWH)**2 #(32 / 8)^2 = 16
patchVectorLen=channel*(patchWH**2) #3 * 64 = 192
embedVectorLen=int(patchVectorLen/2)
#transformer layer hyperparam
head=12
dim_feedforward=embedVectorLen
activation="gelu"
layers=12

#learning param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
decay=0.1