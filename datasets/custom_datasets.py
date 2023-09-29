import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils.utils import set_random_seed
from datasets.cutpast_transformation import *
from PIL import Image
from glob import glob
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

DATA_PATH = './data/'
class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, 'train', "*", "images", "*.JPEG"))
        if count==-1:
            final_length = len(image_files)
        else:
            random.shuffle(image_files)
            final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)

class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-len(self.image_files[:t])):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)

class FakeMVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        self.image_files = glob(os.path.join(root, category, "*.jpeg"))
        if count!=-1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-len(self.image_files[:t])):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1
    def __len__(self):
        return len(self.image_files)

class MVTecDataset_Cutpasted(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count!=-1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)
    

class DataOnlyDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx][0]
        return sample

class HEAD_CT_DATASET(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, 1-self.labels[index]
    
    def __len__(self):
        return len(self.image_files)

class FakeCIFAR10(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1
    def __len__(self):
        return len(self.image_files)

class FakeMNIST(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=6000):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = list(np.load("./Fake_Mnist.npy")[6000*i:6000*(i+1)])
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        # self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image = Image.fromarray((self.image_files[index].transpose(1, 2, 0)*255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target
    def __len__(self):
        return len(self.image_files)

class FakeFashionDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1
    def __len__(self):
        return len(self.image_files)

class Fake_SVHN_Dataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target
    def __len__(self):
        return len(self.image_files)
