import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

class DataNormalizer():

    def __init__(self,
                 c = "data/color/",
                 b = "data/bw/",
                 j = "data/"):

        self.colorpath = c
        self.bwpath = b
        self.jsonpath = j

    def get_json(self):
        data = []
        tot_files = len(os.listdir(self.colorpath))

        for i in range(tot_files):
            Xpath = self.colorpath + "color_"+str(i)+".jpg"
            ypath = self.bwpath + "bw_"+str(i)+".jpg"
            if os.path.exists(Xpath) and os.path.exists(ypath):
                data.append([Xpath,ypath])

        vmc0 = vmc1 = vmc2 = vmbw=0
        stdc0 = stdc1 = stdc2 = stdbw = 0

        pbar = tqdm(total=len(data))



        for d in data:
            imcl = cv.imread(d[0])
            imbw = cv.imread(d[1])

            tmpvmc0 = imcl[:,:,0].sum()/(imcl.shape[0]*imcl.shape[1])
            tmpvmc1 = imcl[:,:,1].sum()/(imcl.shape[0]*imcl.shape[1])
            tmpvmc2 = imcl[:,:,2].sum()/(imcl.shape[0]*imcl.shape[1])

            tmpvmbw = imbw[:,:,0].sum()/(imcl.shape[0]*imcl.shape[1])

            vmc0 += tmpvmc0
            vmc1 += tmpvmc1
            vmc2 += tmpvmc2

            vmbw += tmpvmbw

            stdc0 += np.sqrt(((imcl[:,:,0] - tmpvmc0)**2).sum()/(imcl.shape[0]*imcl.shape[1]))
            stdc1 += np.sqrt(((imcl[:,:,1] - tmpvmc1)**2).sum()/(imcl.shape[0]*imcl.shape[1]))
            stdc2 += np.sqrt(((imcl[:,:,2] - tmpvmc2)**2).sum()/(imcl.shape[0]*imcl.shape[1]))

            stdbw += np.sqrt(((imbw[:,:,0] - tmpvmbw)**2).sum()/(imcl.shape[0]*imcl.shape[1]))
            pbar.update(1)

        pbar.close()

        vmc0 /= len(data)
        vmc1 /= len(data)
        vmc2 /= len(data)
        vmbw /= len(data)

        stdc0 /= len(data)
        stdc1 /= len(data)
        stdc2 /= len(data)
        stdbw /= len(data)


        print("vmc0: ", vmc0,"vmc1: ", vmc1,"vmc2: ", vmc2,"vmbw: ", vmbw)

        import json

        jsondata = {"mean":{"vmc0": vmc0,"vmc1":vmc1,"vmc2":vmc2,"vmbw":vmbw},"std":{"stdc0": stdc0,"stdc1":stdc1,"stdc2":stdc2,"stdbw":stdbw}}
        with open(self.jsonpath + "data.json", 'w', encoding='utf-8') as f:
            json.dump(jsondata, f, ensure_ascii=False, indent=4)