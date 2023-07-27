import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json 
import cv2 as cv
import torchvision  


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(np.array(mean, dtype=np.float32))
        std = torch.as_tensor(np.array(std, dtype=np.float32))
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class CustomDatasetGen(Dataset):
    def __init__(self, insize, outsize, datapath, jsonpath):

        
        self.jsonpath = jsonpath
        self.data = []

        if datapath is not None:

            self.trainA = datapath + "HR/"
            self.trainB = datapath + "LR/"
            self.jsonpath = jsonpath

            for name in os.listdir(self.trainA):
                number = name.split('.')[0][:-2]
                Xpath = self.trainB + number + "LR.jpg"
                ypath = self.trainA + number + "HR.jpg"
                if os.path.exists(Xpath) and os.path.exists(ypath):
                    self.data.append([Xpath,ypath])

        # Opening JSON file
        f = open(self.jsonpath + 'data.json')
        self.jsondata = json.load(f)
        f.close()

        self.insize = insize
        self.outsize = outsize


        self.meanA = [  self.jsondata["mean"]["vmA2"],
                        self.jsondata["mean"]["vmA1"],
                        self.jsondata["mean"]["vmA0"]]
        
        self.stdA = [   self.jsondata["std"]["stdA2"],
                        self.jsondata["std"]["stdA1"],
                        self.jsondata["std"]["stdA0"]]
        
        self.meanB = [  self.jsondata["mean"]["vmB2"],
                        self.jsondata["mean"]["vmB1"],
                        self.jsondata["mean"]["vmB0"]]
        
        self.stdB = [   self.jsondata["std"]["stdB2"],
                        self.jsondata["std"]["stdB1"],
                        self.jsondata["std"]["stdB0"]]

        self.NA = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(
                        mean=self.meanA,
                        std=self.stdA,
                    ),
                ])
        
        self.UnA = NormalizeInverse(
                        mean=self.meanA,
                        std=self.stdA,
                    )
        
        self.NB = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(
                        mean=self.meanB,
                        std=self.stdB,
                    ),
                ])
        
        self.UnB = NormalizeInverse(
                        mean=self.meanB,
                        std=self.stdB,
                    )

        
    def normalize(self, Ximg, yimg):
        # from copy import copy
        # import torchvision

        # xi = copy(Ximg)

        yimg = yimg.astype(np.float64)
        
        yimg[:,:,0] -= self.jsondata["mean"]["vmA0"]
        yimg[:,:,1] -= self.jsondata["mean"]["vmA1"]
        yimg[:,:,2] -= self.jsondata["mean"]["vmA2"]
        
        yimg[:,:,0] = yimg[:,:,0] / self.jsondata["std"]["stdA0"]
        yimg[:,:,1] = yimg[:,:,1] / self.jsondata["std"]["stdA1"]
        yimg[:,:,2] = yimg[:,:,2] / self.jsondata["std"]["stdA2"]


        
        Ximg = Ximg.astype(np.float64)
        
        Ximg[:,:,0] -= self.jsondata["mean"]["vmB0"]
        Ximg[:,:,1] -= self.jsondata["mean"]["vmB1"]
        Ximg[:,:,2] -= self.jsondata["mean"]["vmB2"]
        
        Ximg[:,:,0] = Ximg[:,:,0] / self.jsondata["std"]["stdB0"]
        Ximg[:,:,1] = Ximg[:,:,1] / self.jsondata["std"]["stdB1"]
        Ximg[:,:,2] = Ximg[:,:,2] / self.jsondata["std"]["stdB2"]

        return Ximg, yimg

    def denormalize(self, Ximg, yimg):

        if not Ximg is None:
            Ximg[:,:,0] = Ximg[:,:,0] * self.jsondata["std"]["stdB0"]
            Ximg[:,:,1] = Ximg[:,:,1] * self.jsondata["std"]["stdB1"]
            Ximg[:,:,2] = Ximg[:,:,2] * self.jsondata["std"]["stdB2"]

            Ximg[:,:,0] += self.jsondata["mean"]["vmB0"]
            Ximg[:,:,1] += self.jsondata["mean"]["vmB1"]
            Ximg[:,:,2] += self.jsondata["mean"]["vmB2"]
        
        if not yimg is None:
            yimg[:,:,0] = yimg[:,:,0] * self.jsondata["std"]["stdA0"]
            yimg[:,:,1] = yimg[:,:,1] * self.jsondata["std"]["stdA1"]
            yimg[:,:,2] = yimg[:,:,2] * self.jsondata["std"]["stdA2"]

            yimg[:,:,0] += self.jsondata["mean"]["vmA0"]
            yimg[:,:,1] += self.jsondata["mean"]["vmA1"]
            yimg[:,:,2] += self.jsondata["mean"]["vmA2"]

        return Ximg, yimg

    # def denormalize_loss(self, Ximg, yimg):

    #     if not Ximg is None:
    #         Ximg = Ximg * self.jsondata["std"]["stdbw"]
    #         Ximg += self.jsondata["mean"]["vmbw"]

    #     if not yimg is None:
    #         y1 = (yimg.transpose(1,3)[:,:,:,0] * self.jsondata["std"]["stdc0"])
    #         y2 = (yimg.transpose(1,3)[:,:,:,1] * self.jsondata["std"]["stdc1"])
    #         y3 = (yimg.transpose(1,3)[:,:,:,2] * self.jsondata["std"]["stdc2"])

    #         y1 += self.jsondata["mean"]["vmc0"]
    #         y2 += self.jsondata["mean"]["vmc1"]
    #         y3 += self.jsondata["mean"]["vmc2"]

    #     yimg = torch.stack([y1,y2,y3]).transpose(0,1)
    #     #breakpoint()
    #     return Ximg, yimg

    def tensor_as_img(self, ten):
        ten[ten >= 254] = 254
        ten[ten <= 0] = 0
        ten = ten.astype(np.uint8)
        ten = cv.resize(ten, self.outsize)

        return ten


    def __len__(self):
        return len(self.data)
    
    def prepareImg(self, Ximg, yimg):

        xn = yn = None

        if Ximg is not None:
            #Ximg = cv2.cvtColor(LR_img, cv2.COLOR_BGR2GRAY)
            
            Ximg = cv2.resize(Ximg, self.insize)
            xn = self.NB(torch.tensor(Ximg).permute(2,0,1).float())
            

        if yimg is not None:
            yimg = cv2.resize(yimg, self.outsize)
            yn = self.NA(torch.tensor(yimg).permute(2,0,1).float())
        

        return xn, yn

    def __getitem__(self, idx):
        LR_path, HR_path = self.data[idx]
        Ximg = cv2.imread(LR_path)
        yimg = cv2.imread(HR_path)
        xn, yn = self.prepareImg(Ximg, yimg)
        return xn, yn