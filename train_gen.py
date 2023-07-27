import os
import time
from copy import copy

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import MSELoss

import numpy as np
import cv2 as cv
import lpips
from pytorch_msssim import SSIM, MS_SSIM
from torchsummary import summary



from model_files.resunet import ResNetUNet, ResNetUNetPS
from normalize_gen import DataNormalizerGen
from dataloader_gen import CustomDatasetGen
from utils import parse_argv_train, save_loss



jsonpath, datasetpath, SAVE_PATH, insize, outsize, BATCH, EPOCH, VIS_DEBUG, LOAD_CHKP, COLAB = parse_argv_train()

colorpath = datasetpath + "color/"
bwpath = datasetpath + "bw/"

if COLAB:
    from google.colab.patches import cv2_imshow
    cv.imshow = lambda n,i : cv2_imshow(i)

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
model = ResNetUNetPS().to(DEVICE)


# trainA = "/home/lollo/Documents/tesi/data/take_11_dataset/HR/"
# trainB = "/home/lollo/Documents/tesi/data/take_11_dataset/LR/"
jsonpath = "/home/lollo/Documents/tesi/data/take_11_dataset/"
datasetpath = "/home/lollo/Documents/tesi/data/take_11_dataset/"
#load the custom dattaset and correspondent dataloader
dataset = CustomDatasetGen(insize, outsize, datasetpath, jsonpath)
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

#print model and parameters number
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(model)
print(summary(model, (3,224,224)))
#setup optimizer and loss function
#opt = SGD(model.parameters(), lr=LR)
opt = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
lpips_loss = lpips.LPIPS(net='vgg')
if DEVICE == "cuda":
  lpips_loss.cuda()
LossLPIPS = lpips_loss

LossMSE = MSELoss()

LossMSSSIM = MS_SSIM(data_range=255, size_average=True, channel=3)

LossSSIM = SSIM(data_range=255, size_average=True, channel=3, nonnegative_ssim=True)


epoch = 0

#if set, load the a saved checkpoint
if (LOAD_CHKP != ""):
    chkp_path = LOAD_CHKP

    if ".chkp" not in chkp_path:
        arr = [ x if ".chkp" in x else "0" for x in os.listdir(chkp_path)]
        arr.sort
        chkp_path = chkp_path + arr[-1]
        #epoch = int(arr[-1].split('_')[-1].split('.')[0]) + 1
    #breakpoint()
    checkpoint = torch.load(chkp_path,map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1



#training loop

model.train()

model.freeze_backbone()

loss_dict = {}

for i in range(epoch, EPOCH):
    print("############################### EPOCH ",i,"#####################################################################\n")

    epoch_start = time.time()
    epochLoss = 0
    batchItems = 0
    stop = True
    count = 0
    loss_dict["epoch_"+str(i)] = {}
    #loop thru single batch
    for batch_id, (X,y) in enumerate(data_loader):

        #convert tensors to float and load em to device
        # X = X.float()
        # y = y.float()
        (X,y) = (X.to(DEVICE), y.to(DEVICE))

        #actual trainign lol #####
        predictions = model(X)

        yZeroMax = dataset.UnA(y)
        predZeroMax = dataset.UnA(predictions)

        yNorm = ((yZeroMax*2)/255 - 1)
        predNorm = ((predZeroMax*2)/255 - 1)


        lpips_val = LossLPIPS(predNorm, yNorm).mean()
        mse_val = LossMSE(predictions, y)
        #msssim_val = LossMSSSIM(predZeroMax, yZeroMax)
        ssim_val = LossSSIM(predZeroMax, yZeroMax)

        #breakpoint()
        #print(round(msssim_val.item(),5), " ", round(ssim_val.item(),5))
        loss = lpips_val + 0.75 * ssim_val + 0.05 * mse_val

        opt.zero_grad()
        #breakpoint()
        loss.backward()
        opt.step()
        ##########################

        count+=1
        if VIS_DEBUG != False and count%VIS_DEBUG==0 :
            
            Ximg = dataset.tensor_as_img(dataset.UnB(X)[0].permute(1,2,0).detach().cpu().numpy())
            yimg = dataset.tensor_as_img(dataset.UnA(y)[0].permute(1,2,0).detach().cpu().numpy())
            pred = dataset.tensor_as_img(dataset.UnA(predictions)[0].permute(1,2,0).detach().cpu().numpy())

            cv.imshow("im",np.hstack([Ximg,yimg,pred]))
            cv.imwrite(img_savepath+"img_"+str(i)+"_"+str(count)+".jpg", np.hstack([Ximg,yimg,pred]))
            print("batch_loss: ", round(loss.item(),4))
            cv.waitKey(1)

            loss_dict["epoch_"+str(i)]["batch_"+str(count/VIS_DEBUG)] = round(loss.item(),5)
            save_loss(loss_dict, current_savepath)
            #breakpoint()
        #breakpoint()    
        epochLoss += loss.item()
        batchItems += BATCH
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print("\nepoch_time: ",round(epoch_time,2), " seconds")
    print("epoch_loss: ", round(epochLoss/batchItems,8))
    loss_dict["epoch_"+str(i)]["epochloss"] = epochLoss
    #save checkpoint
    print("[SAVE] saving checkpoint...")
    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': epochLoss/batchItems,
            }, current_savepath + "checkpoint_"+str(i)+".chkp")