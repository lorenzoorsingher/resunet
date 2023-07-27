import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

class DataNormalizerGen():

    def __init__(self,
                 trainA = "data/color/",
                 trainB = "data/bw/",
                 j = "data/"):

        self.trainA = trainA
        self.trainB = trainB
        self.jsonpath = j

    def get_json(self):
        data = []
        tot_files = len(os.listdir(self.trainA))
        #breakpoint()

        for name in os.listdir(self.trainA):
            number = name.split('.')[0][:-2]
            Xpath = self.trainB + number + "LR.jpg"
            ypath = self.trainA + number + "HR.jpg"
            if os.path.exists(Xpath) and os.path.exists(ypath):
                data.append([Xpath,ypath])


        vmA0 = vmA1 = vmA2 = 0
        stdA0 = stdA1 = stdA2 = 0

        vmB0 = vmB1 = vmB2 = 0
        stdB0 = stdB1 = stdB2 = 0

        pbar = tqdm(total=len(data))



        for d in data:
            imA = cv.imread(d[0])
            imB = cv.imread(d[1])

            tmpvmA0 = imA[:,:,0].sum()/(imA.shape[0]*imA.shape[1])
            tmpvmA1 = imA[:,:,1].sum()/(imA.shape[0]*imA.shape[1])
            tmpvmA2 = imA[:,:,2].sum()/(imA.shape[0]*imA.shape[1])

            tmpvmB0 = imB[:,:,0].sum()/(imB.shape[0]*imB.shape[1])
            tmpvmB1 = imB[:,:,1].sum()/(imB.shape[0]*imB.shape[1])
            tmpvmB2 = imB[:,:,2].sum()/(imB.shape[0]*imB.shape[1])

            vmA0 += tmpvmA0
            vmA1 += tmpvmA1
            vmA2 += tmpvmA2

            vmB0 += tmpvmB0
            vmB1 += tmpvmB1
            vmB2 += tmpvmB2

            stdA0 += np.sqrt(((imA[:,:,0] - tmpvmA0)**2).sum()/(imA.shape[0]*imA.shape[1]))
            stdA1 += np.sqrt(((imA[:,:,1] - tmpvmA1)**2).sum()/(imA.shape[0]*imA.shape[1]))
            stdA2 += np.sqrt(((imA[:,:,2] - tmpvmA2)**2).sum()/(imA.shape[0]*imA.shape[1]))

            stdB0 += np.sqrt(((imB[:,:,0] - tmpvmB0)**2).sum()/(imB.shape[0]*imB.shape[1]))
            stdB1 += np.sqrt(((imB[:,:,1] - tmpvmB1)**2).sum()/(imB.shape[0]*imB.shape[1]))
            stdB2 += np.sqrt(((imB[:,:,2] - tmpvmB2)**2).sum()/(imB.shape[0]*imB.shape[1]))

            pbar.update(1)

        pbar.close()

        vmA0 /= len(data)
        vmA1 /= len(data)
        vmA2 /= len(data)

        stdA0 /= len(data)
        stdA1 /= len(data)
        stdA2 /= len(data)

        vmB0 /= len(data)
        vmB1 /= len(data)
        vmB2 /= len(data)

        stdB0 /= len(data)
        stdB1 /= len(data)
        stdB2 /= len(data)


        print("vmA0: ", vmA0,"vmA1: ", vmA1,"vmA2: ", vmA2)
        print("vmB0: ", vmB0,"vmB1: ", vmB1,"vmB2: ", vmB2)

        import json

        jsondata = {"mean":{"vmA0": vmA0,"vmA1":vmA1,"vmA2":vmA2,"vmB0": vmB0,"vmB1":vmB1,"vmB2":vmB2,},
                    "std":{"stdA0": stdA0,"stdA1":stdA1,"stdA2":stdA2,"stdB0": stdB0,"stdB1":stdB1,"stdB2":stdB2,}}
        with open(self.jsonpath + "data.json", 'w', encoding='utf-8') as f:
            json.dump(jsondata, f, ensure_ascii=False, indent=4)