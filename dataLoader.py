import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json 
import cv2 as cv

class CustomDataset(Dataset):
    def __init__(self, insize, outsize, datapath, jsonpath):

        self.path = datapath
        self.colorpath = self.path + "color/"
        self.bwpath = self.path + "bw/"
        self.jsonpath = jsonpath
        self.data = []
        tot_files = len(os.listdir(self.colorpath))

        for i in range(tot_files):
            Xpath = self.colorpath + "color_"+str(i)+".jpg"
            ypath = self.bwpath + "bw_"+str(i)+".jpg"
            if os.path.exists(Xpath) and os.path.exists(ypath):
                self.data.append([Xpath,ypath])
        
        # Opening JSON file
        f = open(self.jsonpath + 'data.json')
        self.jsondata = json.load(f)
        f.close()

        self.insize = insize
        self.outsize = outsize

    def normalize(self, Ximg, yimg):

        yimg = yimg.astype(np.float64)
        
        yimg[:,:,0] -= self.jsondata["mean"]["vmc0"]
        yimg[:,:,1] -= self.jsondata["mean"]["vmc1"]
        yimg[:,:,2] -= self.jsondata["mean"]["vmc2"]
        
        yimg[:,:,0] = yimg[:,:,0] / self.jsondata["std"]["stdc0"]
        yimg[:,:,1] = yimg[:,:,1] / self.jsondata["std"]["stdc1"]
        yimg[:,:,2] = yimg[:,:,2] / self.jsondata["std"]["stdc2"]


        
        Ximg = Ximg.astype(np.float64)

        Ximg -= self.jsondata["mean"]["vmbw"]

        Ximg = Ximg / self.jsondata["std"]["stdbw"]

        return Ximg, yimg

    def denormalize(self, Ximg, yimg):

        if not Ximg is None:
            Ximg = Ximg * self.jsondata["std"]["stdbw"]
            Ximg += self.jsondata["mean"]["vmbw"]
        
        if not yimg is None:
            yimg[:,:,0] = yimg[:,:,0] * self.jsondata["std"]["stdc0"]
            yimg[:,:,1] = yimg[:,:,1] * self.jsondata["std"]["stdc1"]
            yimg[:,:,2] = yimg[:,:,2] * self.jsondata["std"]["stdc2"]

            yimg[:,:,0] += self.jsondata["mean"]["vmc0"]
            yimg[:,:,1] += self.jsondata["mean"]["vmc1"]
            yimg[:,:,2] += self.jsondata["mean"]["vmc2"]

        return Ximg, yimg

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        color_path, bw_path = self.data[idx]

        Ximg = cv2.imread(bw_path)
        yimg = cv2.imread(color_path)

        Ximg = cv2.cvtColor(Ximg, cv2.COLOR_BGR2GRAY)

        Ximg = cv2.resize(Ximg, self.insize)
        yimg = cv2.resize(yimg, self.outsize)

        Ximg, yimg = self.normalize(Ximg, yimg)

        Ximg_tensor = torch.tensor([Ximg])
        yimg_tensor = torch.tensor(yimg)

        Ximg_tensor = Ximg_tensor.permute(0, 2, 1)
        yimg_tensor = yimg_tensor.permute(2, 1, 0)

        return Ximg_tensor, yimg_tensor