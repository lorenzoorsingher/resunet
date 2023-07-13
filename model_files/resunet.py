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

        # self.convBw = nn.Conv2d(
        #     1,
        #     3,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )

        
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
