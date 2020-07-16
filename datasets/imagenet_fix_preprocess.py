import os
import time
import random

import cv2
import numpy as np
import torch

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import get_subclass_dataset

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

IMAGENET_PATH = '~/data/ImageNet'


check = time.time()

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

# remove airliner(1), ambulance(2), parking_meter(18), schooner(22) since similar class exist in CIFAR-10
class_idx_list = list(range(30))
remove_idx_list = [1, 2, 18, 22]
for remove_idx in remove_idx_list:
    class_idx_list.remove(remove_idx)

set_random_seed(0)
train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
Imagenet_set = datasets.ImageFolder(train_dir, transform=transform)
Imagenet_set = get_subclass_dataset(Imagenet_set, class_idx_list)
Imagenet_dataloader = DataLoader(Imagenet_set, batch_size=100, shuffle=True, pin_memory=False)

total_test_image = None
for n, (test_image, target) in enumerate(Imagenet_dataloader):

    if n == 0:
        total_test_image = test_image
    else:
        total_test_image = torch.cat((total_test_image, test_image), dim=0).cpu()

    if total_test_image.size(0) >= 10000:
        break

print (f'Preprocessing time {time.time()-check}')

if not os.path.exists('./Imagenet_fix'):
    os.mkdir('./Imagenet_fix')

check = time.time()
for i in range(10000):
    save_image(total_test_image[i], f'Imagenet_fix/correct_resize_{i}.png')
print (f'Saving time {time.time()-check}')

