from model_files import model
from model_files.resunet import ResNetUNet

import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from dataLoader import CustomDataset
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import os
from copy import copy
import time

#@title Training Loop

import lpips

LOAD_CHKP = False
VIS_DEBUG = True
SAVE_PATH = "/content/drive/MyDrive/tesi/data/lfw/data/checkpoints/"
BATCH = 32
EPOCH = 300
VISUAL = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))


#folders for checkpoints and degub images
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)
current_savepath = SAVE_PATH + "run_"+str(round(time.time()))+"/"
img_savepath = current_savepath + "imgs/"
os.mkdir(current_savepath)
os.mkdir(img_savepath)


#load the model
#ae = model.SimplerAE2().to(DEVICE)
model = ResNetUNet().to(DEVICE)

insize, outsize = ((128,128),(128,128))
#load the custom dattaset and correspondent dataloader
dataset = CustomDataset(insize, outsize)
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

#print model and parameters number
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(model)

#setup optimizer and loss function
opt = SGD(model.parameters(), lr=LR)
lpips_loss = lpips.LPIPS(net='alex')
if DEVICE == "cuda":
  lpips_loss.cuda()
lossFunc = lpips_loss



#if set, load the a saved checkpoint
if (LOAD_CHKP):
    chkp_path = "checkpoints/run_1688680178/checkpoint_8.chkp"
    checkpoint = torch.load(chkp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

# ### load example image (not from dataset)
# Xmple = cv.imread(Xmple_path)
# ymple = cv.imread(ymple_path)
# Xmple = cv.cvtColor(Xmple, cv.COLOR_BGR2GRAY)

# Xmple = cv.resize(Xmple, dataset.insize)
# ymple = cv.resize(ymple, dataset.outsize)

# Xmple, ymple = dataset.normalize(Xmple, ymple)

# Xmple = torch.tensor([[Xmple]])
# ymple = torch.tensor([ymple])

# Xmple = Xmple.permute(0,1, 3, 2)
# ymple = ymple.permute(0,3, 2, 1)
# Xmple = Xmple.float()
# ymple = ymple.float()
# ########################################

#cv.namedWindow("encode_decode_result", cv.WINDOW_NORMAL)

#training loop
model.train()
epoch = 0
for i in range(EPOCH):
    print("############################### EPOCH ",i,"#####################################################################\n")

    epoch_start = time.time()
    epochLoss = 0
    batchItems = 0
    stop = True
    count = 0

    #loop thru single batch
    for batch_id, (X,y) in enumerate(data_loader):

        #convert tensors to float and load em to device
        X = X.float()
        y = y.float()
        (X,y) = (X.to(DEVICE), y.to(DEVICE))

        #actual trainign lol #####
        predictions,_ = model(X)

        loss = lossFunc(predictions, y)

        opt.zero_grad()
        #breakpoint()
        loss.sum().backward()
        opt.step()
        ##########################

        count+=1
        if count%VISUAL==0 and VIS_DEBUG:

            #basically multiply std and add mean for each channel
            Ximg, _ = dataset.denormalize(copy(X[0].detach().transpose(0,2).cpu().numpy()),None)
            pred, yimg = dataset.denormalize(copy(predictions[0].detach().transpose(0,2).cpu().numpy()),copy(y[0].detach().transpose(0,2).cpu().numpy()))

            #tensor to ndarray, resize and gray to bgr to allaw hstacking
            Ximg = Ximg.astype(np.uint8)
            Ximg = cv.resize(Ximg, yimg.shape[:2])
            Ximg = cv.cvtColor(Ximg,cv.COLOR_GRAY2BGR)
            yimg = yimg.astype(np.uint8)
            #some values exceed 254 or are negative (no tanh, sigmoid or similar in net
            #because data is already standardized)
            pred[pred > 254] = 254
            pred[pred < 0] = 0
            pred = pred.astype(np.uint8)

            # #same as above
            # example_pred, _ = ae(Xmple.to(DEVICE))
            # ex_Ximg, _ = dataset.denormalize(copy(Xmple[0].detach().transpose(0,2).cpu().numpy()),None)
            # ex_pred, ex_yimg = dataset.denormalize(copy(example_pred[0].detach().transpose(0,2).cpu().numpy()),copy(ymple[0].detach().transpose(0,2).cpu().numpy()))

            # ex_Ximg = ex_Ximg.astype(np.uint8)
            # ex_Ximg = cv.resize(ex_Ximg, yimg.shape[:2])
            # ex_Ximg = cv.cvtColor(ex_Ximg,cv.COLOR_GRAY2BGR)
            # ex_yimg = ex_yimg.astype(np.uint8)
            # ex_pred[ex_pred > 254] = 254
            # ex_pred[ex_pred < 0] = 0
            # ex_pred = ex_pred.astype(np.uint8)

            cv.imshow("im",np.hstack([Ximg,yimg,pred]))
            #cv.imwrite(img_savepath+"img_"+str(i)+"_"+str(count)+".jpg",np.hstack([ex_Ximg,ex_yimg,ex_pred]))
            print("batch_loss: ", round(loss.sum().item()/BATCH,4))
            cv.waitKey(1)

        epochLoss += loss.sum().item()
        batchItems += BATCH
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print("\nepoch_time: ",round(epoch_time,2), " seconds")
    print("epoch_loss: ", round(epochLoss/batchItems,8))

    #save checkpoint
    print("[SAVE] saving checkpoint...")
    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': epochLoss/batchItems,
            }, current_savepath + "checkpoint_"+str(i)+".chkp")