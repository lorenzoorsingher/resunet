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
from dataLoader import CustomDataset
from utils import parse_argv_inf, save_loss



inpath, jsonpath, size, LOAD_CHKP, COLAB = parse_argv_inf()


if COLAB:
    from google.colab.patches import cv2_imshow
    cv.imshow = lambda n,i : cv2_imshow(i)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))



#load the model
#ae = model.SimplerAE2().to(DEVICE)
model = ResNetUNetPS().to(DEVICE)

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
    epoch = checkpoint['epoch'] + 1



#training loop

model.eval()

dataset = CustomDataset(size, size, None, jsonpath)


if inpath == "":
    inpath = "data/bwme.jpg"

input_img = cv.imread(inpath)

X = dataset.prepareImg(input_img, None)[0].unsqueeze(0)

epoch_start = time.time()

predictions = model(X)

epoch_end = time.time()
epoch_time = epoch_end - epoch_start
print("\nepoch_time: ",round(epoch_time,4), " seconds")

pred = dataset.tensor_as_img(dataset.UnCol(predictions)[0].permute(1,2,0).detach().cpu().numpy())


input_img = cv.resize(input_img, dataset.outsize)
cv.imshow("im",np.hstack([input_img,pred]))

cv.waitKey(0)



