import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json 
import cv2 as cv
import torchvision  

# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = np.array(mean, dtype=np.float32)
#         self.std = np.array(std, dtype=np.float32)

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         t_clone = tensor.clone()
#         for t, m, s in zip(t_clone, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return t_clone

# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = np.array(mean, dtype=np.float32)
#         self.std = np.array(std, dtype=np.float32)

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         t_clone = tensor.clone()
#         for t, m, s in zip(t_clone, self.mean, self.std):
#             t.sub_(m).div_(s)
#             # The normalize code -> t.sub_(m).div_(s)
#         return t_clone

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

class CustomDataset(Dataset):
    def __init__(self, insize, outsize, datapath, jsonpath):

        
        self.jsonpath = jsonpath
        self.data = []

        if datapath is not None:
            self.path = datapath
            self.colorpath = self.path + "color/"
            self.bwpath = self.path + "bw/"

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


        self.meanC = [  self.jsondata["mean"]["vmc2"],
                        self.jsondata["mean"]["vmc1"],
                        self.jsondata["mean"]["vmc0"]]
        
        self.stdC = [   self.jsondata["std"]["stdc2"],
                        self.jsondata["std"]["stdc1"],
                        self.jsondata["std"]["stdc0"]]
        
        self.meanB = [self.jsondata["mean"]["vmbw"]]

        self.stdB = [self.jsondata["std"]["stdbw"]]

        self.NCol = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(
                        mean=self.meanC,
                        std=self.stdC,
                    ),
                ])
        
        self.UnCol = NormalizeInverse(
                        mean=self.meanC,
                        std=self.stdC,
                    )
        
        self.NBw = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=self.meanB,
                std=self.stdB,
            ),
        ])

        self.UnBw = NormalizeInverse(
                        mean=self.meanB,
                        std=self.stdB,
                    )

        
    def normalize(self, Ximg, yimg):
        # from copy import copy
        # import torchvision

        # xi = copy(Ximg)

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

    def denormalize_loss(self, Ximg, yimg):

        if not Ximg is None:
            Ximg = Ximg * self.jsondata["std"]["stdbw"]
            Ximg += self.jsondata["mean"]["vmbw"]

        if not yimg is None:
            y1 = (yimg.transpose(1,3)[:,:,:,0] * self.jsondata["std"]["stdc0"])
            y2 = (yimg.transpose(1,3)[:,:,:,1] * self.jsondata["std"]["stdc1"])
            y3 = (yimg.transpose(1,3)[:,:,:,2] * self.jsondata["std"]["stdc2"])

            y1 += self.jsondata["mean"]["vmc0"]
            y2 += self.jsondata["mean"]["vmc1"]
            y3 += self.jsondata["mean"]["vmc2"]

        yimg = torch.stack([y1,y2,y3]).transpose(0,1)
        #breakpoint()
        return Ximg, yimg

    def tensor_as_img(self, ten):
        ten[ten >= 254] = 254
        ten[ten <= 0] = 0
        ten = ten.astype(np.uint8)
        ten = cv.resize(ten, self.outsize)

        return ten


    def __len__(self):
        return len(self.data)
    
    def prepareImg(self, bw_img, color_img):

        xn = yn = None

        if bw_img is not None:
            Ximg = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)
            Ximg = cv2.resize(Ximg, self.insize)
            xn = self.NBw(torch.tensor(Ximg).unsqueeze(0).float())

        if color_img is not None:
            yimg = cv2.resize(color_img, self.outsize)
            yn = self.NCol(torch.tensor(yimg).permute(2,0,1).float())

        return xn, yn

    def __getitem__(self, idx):
        color_path, bw_path = self.data[idx]
        Ximg = cv2.imread(bw_path)
        yimg = cv2.imread(color_path)
        xn, yn = self.prepareImg(Ximg, yimg)
        return xn, yn