"""
created by: Donghyeon Won
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from resnext_wsl import *
from models import *


class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        protest = self.label_frame.iloc[idx, 1:2].to_numpy().astype('float')
        violence = self.label_frame.iloc[idx, 2:3].to_numpy().astype('float')
        visattr = self.label_frame.iloc[idx, 3:].to_numpy().astype('float')
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}

        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample

class FinalLayer50(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer50, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("before fc",x.shape)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class FinalLayer18(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer18, self).__init__()
        self.fc = nn.Linear(512, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("before fc",x.shape)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class FinalLayer34(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer34, self).__init__()
        self.fc = nn.Linear(512, 12)  # did not edit value yet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("before fc",x.shape)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out
class FinallayerDensenet161(nn.Module):
    def __init__(self):
        super(FinallayerDensenet161, self).__init__()
        self.classifier = nn.Linear(2208, 12)  # original = 1024
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("before fc",x.shape)
        out = self.classifier(x)
        out = self.sigmoid(out)
        return out

        

def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def modified_eff_resnet():
    model = resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 
    model.fc = FinalLayer50()
    return model

def modified_densenet161():
    model = models.densenet161(pretrained = True)
    model.classifier = FinallayerDensenet161()
    return model
    

def modified_resnet50():
    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained = True)
    model.fc = FinalLayer50()

    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False


    return model

def modified_resnet18():
    # model = ResNet18
    model = models.resnet18(pretrained=True)
    model.fc = FinalLayer18()

    return model

def modified_resnet34():
    # model = ResNet18
    model = models.resnet34(pretrained=True)
    model.fc = FinalLayer34()

    return model
def modified_inception():
    model_ft = models.inception_v3(pretrained=True)
    
    feature_extract = True
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs_Aux = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs_Aux, 12)
    # Handle the primary net
    num_ftrs_Linear = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs_Linear,12)
    input_size = 299
    return model_ft

def modified_squeeznet():
    model_ft = models.squeezenet1_0(pretrained=True)
    feature_extract = True
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, 12, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = 12
    input_size = 224

    return model_ft

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count

class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
