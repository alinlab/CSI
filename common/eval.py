from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C

from datasets import set_dataset_count, mvtecad_dataset, get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import get_loader_unique_label

P = parse_args()

normal_labels = None
if P.normal_labels:
    normal_labels = [int(num) for num in P.normal_labels.split(',')]
    print("normal_labels: ", normal_labels)

cls_list = get_superclass_list(P.dataset)
anomaly_labels = [elem for elem in cls_list if elem not in normal_labels]

### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Initialize dataset ###
ood_eval = P.mode == 'ood_pre'

if P.dataset == 'imagenet' and ood_eval:
    P.batch_size = 1
    P.test_batch_size = 1

if P.image_size==32:
    image_size_ = (32, 32, 3)
else:
    image_size_ = (224, 224, 3)

if P.dataset=="MVTecAD":
    train_set, test_set, image_size, n_classes = mvtecad_dataset(P=P, category=P.one_class_idx, root = "./mvtec_anomaly_detection",  image_size=image_size_)
else:
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, download=True, image_size=image_size_, labels=normal_labels)
P.image_size = image_size
P.n_classes = n_classes

print("full test set:", len(test_set))
print("full train set:", len(train_set))


full_test_set = deepcopy(test_set)  # test set of full classes
if P.dataset=='mvtec-high-var' or P.dataset=="MVTecAD" or P.dataset=="WBC" or P.dataset=='cifar10-versus-100' or P.dataset=='cifar100-versus-10':
    train_set = set_dataset_count(train_set, count=P.main_count)
    test_set = get_subclass_dataset(P, test_set, classes=[0])
else:
    train_set = get_subclass_dataset(P, train_set, classes=normal_labels, count=P.main_count)
    test_set = get_subclass_dataset(P, test_set, classes=normal_labels)
        
print("number of normal test set:", len(test_set))
print("number of normal train set:", len(train_set))

kwargs = {'pin_memory': False, 'num_workers': 4}

train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)


print("len train_set", len(train_set))
print("len test_set", len(test_set))

print("Unique labels(test_loader):", get_loader_unique_label(test_loader))
print("Unique labels(train_loader):", get_loader_unique_label(train_loader))


P.ood_dataset = anomaly_labels
if P.dataset=="MVTecAD" or P.dataset=="mvtec-high-var" or P.dataset=='cifar10-versus-100' or P.dataset=='cifar100-versus-10':
    P.ood_dataset = [1]
print("P.ood_dataset",  P.ood_dataset)

ood_test_loader = dict()
for ood in P.ood_dataset:
    ood_test_set = get_subclass_dataset(P, full_test_set, classes=ood)
    ood = f'one_class_{ood}'
    print(f"testset anomaly(class {ood}):", len(ood_test_set))
    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    print("Unique labels(ood_test_loader):", get_loader_unique_label(ood_test_loader[ood]))
 

print("train loader batchs", len(train_loader))
print("train_set:", len(train_set))
### Initialize model ###

simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

P.K_shift = 2
model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
model = C.get_shift_classifer(model, P.K_shift).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if P.load_path is not None:
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)
