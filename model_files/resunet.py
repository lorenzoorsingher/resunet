from torchvision import models
from torch import Tensor

import torch
import torch.nn as nn

class UpBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: bool = False
    ) -> None:
        super().__init__()

        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False),
                nn.Conv2d(  inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)
                )
            self.conv1 = nn.ConvTranspose2d(
                    inplanes,
                    planes,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
        else:
            self.upsample = None
            self.conv1 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class UpBlockPS(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: bool = False
    ) -> None:
        super().__init__()

        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False),
                nn.Conv2d(  inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)
                )            
            self.pshuffle1 = nn.PixelShuffle(2)
        else:
            self.upsample = None
            self.conv1 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if self.upsample is not None:
            out = self.pshuffle1(x)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetUNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')

        self.layer1 = nn.Sequential(
                UpBlock(512,256,True),
                UpBlock(256,256,False)
        )

        self.layer2 = nn.Sequential(
                UpBlock(512,128,True),
                UpBlock(128,128,False)
        )

        self.layer3 = nn.Sequential(
                UpBlock(256,64,True),
                UpBlock(64,64,False)
        )
        
        self.out = nn.Sequential(

            nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                    64,
                    3,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                )
        )
        
    def forward(self, x):
        
        #breakpoint()
        x = x.expand(x.shape[0],3,x.shape[2],x.shape[3])
        #breakpoint()
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x1 = self.base_model.maxpool(x)

        x2 = self.base_model.layer1(x1)
        x3 = self.base_model.layer2(x2)
        x4 = self.base_model.layer3(x3)
        x = self.base_model.layer4(x4)

        x = self.layer1(x)

        x = torch.cat((x,x4),1)
        x = self.layer2(x)
        x = torch.cat((x,x3),1)
        x = self.layer3(x)
        x = torch.cat((x,x2),1)
        out = self.out(x)

        return out
    
    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
            self.base_model.train(False)

    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
            self.base_model.train(True)

class ResNetUNetPS(nn.Module):

    def __init__(self,rgb_in=True, lastskip = False):
        super().__init__()
        
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')

        self.layer1 = nn.Sequential(
                UpBlock(512,256,True),
                UpBlock(256,256,False)
        )

        self.layer2 = nn.Sequential(
                UpBlockPS(512,128,True),
                UpBlockPS(128,128,False)
        )

        self.layer3 = nn.Sequential(
                UpBlockPS(256,64,True),
                UpBlockPS(64,64,False)
        )

        self.layer4 = nn.Sequential(
                UpBlockPS(128,32,True),
                UpBlockPS(32,32,False)
        )
        
        self.layer5 = nn.Sequential(
                UpBlockPS(96,24,True),
                UpBlockPS(24,24,False)
        )

        self.outUp = nn.Sequential(
                UpBlockPS(24,6,True),
                UpBlockPS(6,6,False)
        )

        self.rgb_in = rgb_in
        self.lastskip = lastskip
        if lastskip:
            if self.rgb_in:
                last_layer_c = 27
            else:
                last_layer_c = 25
        else:
            last_layer_c = 24
        last_layer_c = 6
        self.out = nn.Sequential(
            nn.Conv2d(
                    last_layer_c,
                    3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
        )

        

        
    def forward(self, x):
        
        #breakpoint()
        if self.rgb_in:
            xog = x
        else:
            x = x.expand(x.shape[0],3,x.shape[2],x.shape[3])
            xog = x[:,0,:,:].unsqueeze(1)
        #breakpoint()
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x0 = self.base_model.relu(x)
        x1 = self.base_model.maxpool(x0)

        x2 = self.base_model.layer1(x1)
        x3 = self.base_model.layer2(x2)
        x4 = self.base_model.layer3(x3)
        x5 = self.base_model.layer4(x4)

        x = self.layer1(x5)
        x = torch.cat((x,x4),1)
        x = self.layer2(x)
        x = torch.cat((x,x3),1)
        x = self.layer3(x)
        x = torch.cat((x,x2),1)
        x = self.layer4(x)
        x = torch.cat((x,x0),1)
        x = self.layer5(x)
        if self.lastskip:
            x = torch.cat((x,xog),1)

        x = self.outUp(x)
        
        out = self.out(x)

        return out
    
    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
            self.base_model.train(False)

    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
            self.base_model.train(True)
