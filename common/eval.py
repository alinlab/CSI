from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import mvtecad_dataset, get_dataset, get_superclass_list, get_subclass_dataset

P = parse_args()

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
if P.dataset=="MVTecAD":
    train_set, test_set, image_size, n_classes = mvtecad_dataset(P=P, category=P.one_class_idx, root = "./mvtec_anomaly_detection")
else:
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, eval=ood_eval, download=True)

P.image_size = image_size
P.n_classes = n_classes
print("full test set:", len(test_set))

if P.one_class_idx is not None:
    cls_list = get_superclass_list(P.dataset)
    P.n_superclasses = len(cls_list)
    full_test_set = deepcopy(test_set)  # test set of full classes
    if P.dataset=="MVTecAD":
        test_set = get_subclass_dataset(test_set, classes=0)
    else:
        train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx], count=P.main_count)
        test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])

print("normal test set:", len(test_set))

kwargs = {'pin_memory': False, 'num_workers': 4}

train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if (P.ood_dataset is None) and (P.dataset!="MVTecAD"):
    if P.one_class_idx is not None:
        P.ood_dataset = list(range(P.n_superclasses))
        P.ood_dataset.pop(P.one_class_idx)
    elif P.dataset == 'cifar10':
        P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
    elif P.dataset == 'imagenet':
        P.ood_dataset = ['cub', 'stanford_dogs', 'flowers102', 'places365', 'food_101', 'caltech_256', 'dtd', 'pets']

if P.dataset=="MVTecAD":
    P.ood_dataset = [1]
ood_test_loader = dict()
for ood in P.ood_dataset:
    if ood == 'interp':
        ood_test_loader[ood] = None  # dummy loader
        continue

    if P.one_class_idx is not None:
        ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
        ood = f'one_class_{ood}'  # change save name
    else:
        ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, eval=ood_eval, download=True)
    print(f"testset anomaly(class {ood}):", len(ood_test_set))

    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
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
