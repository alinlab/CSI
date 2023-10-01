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
from datasets.custom_datasets import *
from torch.utils.data import ConcatDataset

import medmnist
from medmnist import INFO, Evaluator

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'


CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class
MNIST_SUPERCLASS = list(range(10))
SVHN_SUPERCLASS = list(range(10))
FashionMNIST_SUPERCLASS = list(range(10))
MVTecAD_SUPERCLASS = list(range(2))
HEAD_CT_SUPERCLASS = list(range(2))
MVTEC_HV_SUPERCLASS = list(range(2))
breastmnist_SUPERCLASS = list(range(2))
CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

CLASS_NAMES = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",]

def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def mvtecad_dataset(P, category, root = "./mvtec_anomaly_detection", image_size=(224, 224, 3)):
    # image_size = (224, 224, 3)
    n_classes = 2
    categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
    train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    
    test_ds_mvtech = MVTecDataset(root=root, train=False, category=categories[category], transform=test_transform, count=-1)
    train_ds_mvtech_normal = MVTecDataset(root=root, train=True, category=categories[category], transform=train_transform, count=P.main_count)
    
    print("test_ds_mvtech shapes: ", test_ds_mvtech[0][0].shape)
    print("train_ds_mvtech_normal shapes: ", train_ds_mvtech_normal[0][0].shape)
    
    return  train_ds_mvtech_normal, test_ds_mvtech, image_size, n_classes
        
    

def get_exposure_dataloader(P, batch_size = 64, image_size=(224, 224, 3),
                            base_path = './tiny-imagenet-200', fake_root="./MvTechAD", root="./mvtec_anomaly_detection" ,count=-1, cls_list=None):
    categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
    if P.dataset=='head-ct' or P.dataset=='breastmnist' or  P.dataset=='mnist' or P.dataset=='fashion-mnist':
        tiny_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ]) 
    else:
        tiny_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ])

    fake_count = int(P.fake_data_percent*count)
    tiny_count = int((1-(P.fake_data_percent+P.cutpast_data_percent))*count)
    cutpast_count = int(P.cutpast_data_percent*count)
    if (fake_count+tiny_count+cutpast_count)!=count:
        tiny_count += (count - (cutpast_count+fake_count+tiny_count))
    print("fake_count, tiny_count, cutpast_count", fake_count, tiny_count, cutpast_count)
    if P.dataset == "MVTecAD":
        fake_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_transform_cutpasted = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
        ])

        
        imagenet_exposure = ImageNetExposure(root=base_path, count=tiny_count, transform=tiny_transform)
        train_ds_mvtech_fake = FakeMVTecDataset(root=fake_root, train=True, category=categories[P.one_class_idx], transform=fake_transform, count=fake_count)
        train_ds_mvtech_cutpasted = MVTecDataset_Cutpasted(root=root, train=True, category=categories[P.one_class_idx], transform=train_transform_cutpasted, count=cutpast_count)
        print("number of fake data:", len(train_ds_mvtech_fake), 'shape:', train_ds_mvtech_fake[0][0].shape)
        print("number of tiny data:", len(imagenet_exposure), 'shape:', imagenet_exposure[0][0].shape)
        print("number of cutpasted data:", len(train_ds_mvtech_cutpasted), 'shape:', train_ds_mvtech_cutpasted[0][0].shape)
        exposureset = torch.utils.data.ConcatDataset([train_ds_mvtech_fake, imagenet_exposure, train_ds_mvtech_cutpasted])

        print("number of exposure:", len(exposureset))
        train_loader = DataLoader(exposureset, batch_size = batch_size)
    else:
        if P.dataset=='head-ct' or P.dataset=='breastmnist' or P.dataset=='mnist' or P.dataset=='fashion-mnist':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation((90, 270)),
                CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
            ])
        elif P.dataset=='svhn-10':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomRotation((90, 270)),
                CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
            ])
        else:
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
            ])
        cutpast_train_set, _, _, _ = get_dataset(P, dataset=P.dataset, download=True, image_size=image_size, train_transform_cutpasted=train_transform_cutpasted)
        if P.dataset=='head-ct':
            cutpast_train_set = set_dataset_count(cutpast_train_set, count=cutpast_count)
        else:
            if P.high_var:
                print("cls_list", cls_list)
                cutpast_train_set = get_subclass_dataset(P, cutpast_train_set, classes=cls_list, count=cutpast_count)
            else:
                cutpast_train_set = get_subclass_dataset(P, cutpast_train_set, classes=cls_list[P.one_class_idx], count=cutpast_count)
                cls_list = [P.one_class_idx]

        if P.dataset!="mvtec-high-var":       
            cutpast_train_set.transform = train_transform_cutpasted
        # cutpast_train_set = DataOnlyDataset(cutpast_train_set)
        imagenet_exposure = ImageNetExposure(root=base_path, count=tiny_count, transform=tiny_transform)
        if P.dataset=="cifar10":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0],image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root='./CIFAR10-Fake/'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))            
            train_ds_cifar10_fake = FakeCIFAR10(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_cifar10_fake) > 0:
                print("number of fake data:", len(train_ds_cifar10_fake), "shape:", train_ds_cifar10_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_cifar10_fake, imagenet_exposure])
        elif P.dataset=="svhn-10":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0],image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root='./SVHN-Fake/'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))            
            train_ds_svhn_fake = Fake_SVHN_Dataset(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_svhn_fake) > 0:
                print("number of fake data:", len(train_ds_svhn_fake), "shape:", train_ds_svhn_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_svhn_fake, imagenet_exposure])
        
        elif P.dataset=="mnist":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0],image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))                  
            fake_root = './MNIST-Fake/'
            train_ds_mnist_fake = FakeMNIST(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_mnist_fake) > 0:
                print("number of fake data:", len(train_ds_mnist_fake), "shape:", train_ds_mnist_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_mnist_fake, imagenet_exposure])
        elif P.dataset=="fashion-mnist":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0],image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            fake_root='./Fashion-Fake/'
            train_ds_fmnist_fake = FakeFashionDataset(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_fmnist_fake) > 0:
                print("number of fake data:", len(train_ds_fmnist_fake), "shape:", train_ds_fmnist_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_fmnist_fake, imagenet_exposure])
        else:
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, imagenet_exposure])
        
        if len(cutpast_train_set) > 0:
            print("number of cutpast data:", len(cutpast_train_set), 'shape:', cutpast_train_set[0][0].shape)
        print("number of tiny data:", len(imagenet_exposure), 'shape:', imagenet_exposure[0][0].shape)
        print("number of exposure:", len(exposureset))
        train_loader = DataLoader(exposureset, batch_size = batch_size)
    return train_loader


def get_breastmnist_test(normal_class_indx, path, transform):
    data_flag = 'breastmnist'
    BATCH_SIZE = 64
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    test_dataset = DataClass(split='test', transform=transform, download=download)

    test_dataset.labels = test_dataset.labels.squeeze()

    '''
    classes = [normal_class_indx]
    normal_indices = []
    for idx, (_, tgt) in enumerate(test_dataset):
        if tgt in classes:
            normal_indices.append(idx)
    anomaly_indices = [i for i in range(len(test_dataset)) if i not in normal_indices]
    for i in anomaly_indices:
        test_dataset.labels[i] = 1
    for i in normal_indices:
        test_dataset.labels[i] = 0
    '''
    return test_dataset
    

def get_breastmnist_train(anomaly_class_indx, path, transform):
    data_flag = 'breastmnist'
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=transform, download=download)
    train_dataset.labels = train_dataset.labels.squeeze()

    '''
    classes = [anomaly_class_indx]
    normal_indices = []

    for idx, (_, tgt) in enumerate(train_dataset):
        if tgt in classes:
            normal_indices.append(idx)

    for i in normal_indices:
        train_dataset.labels[i] = 0

    train_dataset = Subset(train_dataset, normal_indices)
    '''
    return train_dataset


def get_dataset(P, dataset, test_only=False, image_size=(32, 32, 3), download=False, eval=False, train_transform_cutpasted=None):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'head-ct':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        labels_df = pd.read_csv('./head-ct/labels.csv')
        labels = np.array(labels_df[' hemorrhage'].tolist())
        images = np.array(sorted(glob('./head-ct/head_ct/head_ct/*.png')))
        np.random.seed  (1225)
        indicies = np.random.permutation(100)
        train_true_idx, test_true_idx = indicies[:75], indicies[75:]
        train_false_idx, test_false_idx = indicies[:75] + 100, indicies[75:] + 100
        train_idx, test_idx = train_true_idx, np.concatenate((test_true_idx, test_false_idx, train_false_idx))

        train_image, train_label = images[train_idx], labels[train_idx]
        test_image, test_label = images[test_idx], labels[test_idx]

        print("train_image.shape, test_image.shape: ", train_image.shape, test_image.shape)
        print("train_label.shape, test_label.shape: ", train_label.shape, test_label.shape)

        train_set = HEAD_CT_DATASET(image_path=train_image, labels=train_label, transform=train_transform)
        test_set = HEAD_CT_DATASET(image_path=test_image, labels=test_label, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'breastmnist':
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_set = get_breastmnist_test(normal_class_indx=P.one_class_idx, path='./data/', transform=transform)
        train_set = get_breastmnist_train(anomaly_class_indx=P.one_class_idx, path='./data/', transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'mvtec-high-var':
        n_classes = 2
        train_dataset = []
        test_dataset = []
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        for class_idx in labels:
            if train_transform_cutpasted:
                train_dataset.append(MVTecDataset_High_VAR('./', class_name=CLASS_NAMES[class_idx], is_train=True, transform=train_transform_cutpasted))
            else:
                train_dataset.append(MVTecDataset_High_VAR('./', class_name=CLASS_NAMES[class_idx], is_train=True))
            test_dataset.append(MVTecDataset_High_VAR('./', class_name=CLASS_NAMES[class_idx], is_train=False))

        train_set = ConcatDataset(train_dataset)
        test_set = ConcatDataset(test_dataset)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
        print("len(test_dataset), len(train_dataset)", len(test_dataset), len(train_dataset))
    elif dataset == 'fashion-mnist':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.FashionMNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar100':
        # image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'mnist':
        # image_size = (32, 32, 1)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'svhn-10':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_pil':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_pil':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'svhn-10':
        return SVHN_SUPERCLASS
    elif dataset == 'breastmnist':
        return breastmnist_SUPERCLASS
    elif dataset == 'MVTecAD':
        return MVTecAD_SUPERCLASS
    elif dataset == 'head-ct':
        return HEAD_CT_SUPERCLASS
    elif dataset == 'mvtec-high-var':
        return MVTEC_HV_SUPERCLASS
    elif dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'fashion-mnist':
        return FashionMNIST_SUPERCLASS
    elif dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(P, dataset, classes, count=-1):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    try:        
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        # SVHN
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)
   

    dataset = Subset(dataset, indices)
    if count==-1:
        pass
    elif len(dataset)>count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num*len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset

def set_dataset_count(dataset, count=-1):
    if count==-1:
        pass
    elif len(dataset)>count:
        dataset = Subset(dataset, [i for i in range(count)])
    else:
        num = int(count / len(dataset))
        remainding = (count - num*len(dataset))
        trnsets = [dataset for i in range(num)]
        dataset = Subset(dataset, [i for i in range(remainding)])
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset

def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


