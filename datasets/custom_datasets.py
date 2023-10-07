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
import rasterio
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import subprocess
from tqdm import tqdm
import requests

import shutil
import random
import zipfile
import time

CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
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
        print("category MVTecDataset:", category)
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
        print("category FakeMVTecDataset:", category)
        self.image_files = glob(os.path.join(root, category, "*.jpeg"))
        if count!=-1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
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
        print("category MVTecDataset_Cutpasted:", category)
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





class MVTecDataset_High_VAR(Dataset):
    def __init__(
        self,
        dataset_path="./mvtec_anomaly_detection",
        class_name="bottle",
        is_train=True,
        resize=256,
        cropsize=224,
        transform=None,
    ):
        assert class_name in CLASS_NAMES, "class_name: {}, should be in {}".format(
            class_name, CLASS_NAMES
        )
        print("class_name MVTecDataset_High_VAR:", class_name)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        self.dataset_path = os.path.join(dataset_path, "mvtec_anomaly_detection")
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        

        # set transforms
        if transform:
            self.transform_x = transform
        else:
            self.transform_x = transforms.Compose(
                [
                    transforms.Resize(resize, Image.ANTIALIAS),
                    transforms.CenterCrop(cropsize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST), transforms.CenterCrop(cropsize), transforms.ToTensor()]
        )

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)
        return x, y

    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"
        return list(x), list(y), list(mask)


class FakeCIFAR100(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=2500):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = list(np.load("./cifar100_training_gen_data.npy")[2500*i:2500*(i+1)])
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        # self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image = Image.fromarray(self.image_files[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1
    def __len__(self):
        return len(self.image_files)



class UCSDDataset(Dataset):
    def __init__(self, root, dataset='ped1', is_normal=True, transform=None, target_transform=None, download=False):
        self.root = os.path.join(root, 'UCSD_Anomaly_Dataset.v1p2')
        self.is_normal = is_normal
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        # download not supported

        if self.dataset == 'ped1':
            base_dir = 'UCSDped1'
        if self.dataset == 'ped2':
            base_dir = 'UCSDped2'

        if not self.is_normal:
            sub_dir = 'Test'
        else:
            sub_dir = 'Train'

        video_dir = glob(os.path.join(self.root, base_dir, sub_dir, sub_dir+'*'))
        self.video_dir = sorted([x for x in video_dir if re.fullmatch('.*\d\d\d', x)])
        self.videos_len = []
        self.images_dir = []
        for video in self.video_dir:
            images = list(sorted(glob(os.path.join(video, "*.tif"))))
            self.images_dir += images
            self.videos_len.append(len(images))
        self.num_samples = len(self.images_dir)
        self.labels = self._gather_labels()



    def __getitem__(self, index):
        with rasterio.open(self.images_dir[index]) as image:
            image_array = image.read()
            # torch.Size([238, 1, 158])
            image = transforms.ToPILImage(mode='RGB')(
                transforms.ToTensor()(image_array).permute(1, 2, 0).repeat(3, 1, 1)
            )
        if self.transform:
            image = self.transform(image)
        label = self.get_label(index)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _gather_labels(self):
        if self.is_normal:
            return None
        if self.dataset == 'ped1':
            base_dir = 'UCSDped1'
        if self.dataset == 'ped2':
            base_dir = 'UCSDped2'
        
        with open(os.path.join(self.root, base_dir, 'Test', f'{base_dir}.m'), 'r') as file:
            lines = file.readlines()

        annotations = []


        video_index = 0
        # Iterate over the lines
        for line in lines:
            # Use regular expressions to extract the frame ranges
            matches = re.findall(r'(\d+:\d+)', line)
            if len(matches) == 0:
                continue

            frame_mask = np.zeros(self.videos_len[video_index], dtype=bool)
            for match in matches:
                start, end = map(int, match.split(':'))
                frame_mask[start-1:end] = True

            annotations.append(frame_mask)
            video_index += 1
        annotations = np.concatenate(annotations)
        return annotations

    def get_label(self, index):
        if self.is_normal:
            label = 0
        else:
            label = 1 if self.labels[index] else 0
            

        return label

    def __len__(self):
        return len(self.images_dir)




class TumorDetection(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True, count=None):
        self._download_and_extract()
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join( './MRI', "Training", "notumor", "*.jpg")
            )
        else:
          image_files = glob(os.path.join( './MRI', "Testing", "*", "*.jpg"))
          normal_image_files = glob(os.path.join( './MRI', "./Testing", "notumor", "*.jpg"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.image_files = image_files

        if count is not None:
            if count > len(self.image_files):
                self.image_files = self._oversample(count)
            else:
                self.image_files  = self._undersample(count)

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
    
    def _download_and_extract(self):
        google_id = '1AOPOfQ05aSrr2RkILipGmEkgLDrZCKz_'
        file_path = os.path.join('./MRI', 'Training')

        if os.path.exists(file_path):
            return

        if not os.path.exists('./MRI'):
            os.makedirs('./MRI')

        if not os.path.exists(file_path):
            subprocess.run(['gdown', google_id, '-O', './MRI/archive(3).zip'])
        
        with zipfile.ZipFile("./MRI/archive(3).zip", 'r') as zip_ref:
            zip_ref.extractall("./MRI/")

        os.rename(  "./MRI/Training/glioma", "./MRI/Training/glioma_tr")
        os.rename(  "./MRI/Training/meningioma", "./MRI/Training/meningioma_tr")
        os.rename(  "./MRI/Training/pituitary", "./MRI/Training/pituitary_tr")
        
        shutil.move("./MRI/Training/glioma_tr","./MRI/Testing")
        shutil.move("./MRI/Training/meningioma_tr","./MRI/Testing")
        shutil.move("./MRI/Training/pituitary_tr","./MRI/Testing")


    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((256, 256))
        
        if self.transform:
            image = self.transform(image)
        
        if "notumor" in os.path.dirname(image_file):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


    def _oversample(self, count):
        num_extra_samples = count - len(self.image_files)
        extra_image_files = [random.choice(self.image_files) for _ in range(num_extra_samples)]

        return self.image_files + extra_image_files

    def _undersample(self, count):
        indices = random.sample(range(len(self.image_files)), count)
        new_image_files = [self.image_files[idx] for idx in indices]

        return new_image_files


class AdaptiveExposure(Dataset):
    def __init__(self, root, transform, count=None):
        super(AdaptiveExposure, self).__init__()
        self.root = root
        self.image_files = glob(os.path.join(root, '**', "*.png"), recursive=True)
        self.transform = transform
        if count is not None:
            if count > len(self.image_files):
                self.image_files = self._oversample(count)
            else:
                self.image_files = self._undersample(count)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((256, 256))
        
        if self.transform:
            image = self.transform(image)
        
        return image, 1

    def _oversample(self, count):
        num_extra_samples = count - len(self.image_files)
        extra_image_files = [random.choice(self.image_files) for _ in range(num_extra_samples)]

        return self.image_files + extra_image_files

    def _undersample(self, count):
        indices = random.sample(range(len(self.image_files)), count)
        new_image_files = [self.image_files[idx] for idx in indices]

        return new_image_files