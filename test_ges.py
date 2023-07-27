from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np

from normalize_gen import DataNormalizerGen
from dataloader_gen import CustomDatasetGen

trainA = "/home/lollo/Documents/tesi/data/take_11_dataset/HR/"
trainB = "/home/lollo/Documents/tesi/data/take_11_dataset/LR/"
jsonpath = "/home/lollo/Documents/tesi/data/take_11_dataset/"
datasetpath = "/home/lollo/Documents/tesi/data/take_11_dataset/"

# dn = DataNormalizerGen(trainA, trainB, jsonpath)
# dn.get_json()

insize = (224,224)
outsize = (224,224)
BATCH = 5

dataset = CustomDatasetGen(insize, outsize, datasetpath, jsonpath)
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

for i in range(5):
    print("############################### EPOCH ",i,"#####################################################################\n")

    
    #loop thru single batch
    for batch_id, (X,y) in enumerate(data_loader):

        
            
        Ximg = dataset.tensor_as_img(dataset.UnB(X)[0].permute(1,2,0).numpy())
        yimg = dataset.tensor_as_img(dataset.UnA(y)[0].permute(1,2,0).numpy())
        #pred = dataset.tensor_as_img(dataset.UnCol(predictions)[0].permute(1,2,0).detach().cpu().numpy())

        cv.imshow("im",np.hstack([Ximg,yimg]))
        cv.waitKey(0)

