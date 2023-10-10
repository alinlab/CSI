'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.transform_layers import NormalizeLayer
from torch.nn.utils import spectral_norm
from torchvision import models
import timm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Pretrain_VIT(BaseModel):
    def __init__(self, num_classes=10):
        last_dim = 768
        super(Pretrain_VIT, self).__init__(last_dim, num_classes)

        self.in_planes = 64
        self.last_dim = 768

        mu = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cuda()
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cuda()
        self.norm = lambda x: ( x - mu ) / std
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.backbone.head = torch.nn.Identity()
        i = 0
        num = 76
        for param in self.backbone.parameters():
            if i<num:
                param.requires_grad = False
            i+=1
      
    def penultimate(self, x, all_features=False):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def VIT_Pretrain(num_classes):
    return Pretrain_VIT(num_classes=num_classes)
