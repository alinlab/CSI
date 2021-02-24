# CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances

Official PyTorch implementation of
["**CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances**"](
https://arxiv.org/abs/2007.08176) (NeurIPS 2020) by
[Jihoon Tack*](https://jihoontack.github.io),
[Sangwoo Mo*](https://sites.google.com/view/sangwoomo),
[Jongheon Jeong](https://sites.google.com/view/jongheonj),
and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).

<p align="center">
    <img src=figures/shifting_transformations.png width="900"> 
</p>

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboard 2.0+
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 
- [apex](https://github.com/NVIDIA/apex) == 0.1
- [diffdist](https://github.com/ag14774/diffdist) == 0.1 

### Datasets 
For CIFAR, please download the following datasets to `~/data`.
* [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz),
[ImageNet_resize](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing),
[ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing)

For ImageNet-30, please download the following datasets to `~/data`.
* [ImageNet-30-train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view),
[ImageNet-30-test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view)
* [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/),
[Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/),
[Oxford flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/),
[Food-101](https://www.kaggle.com/dansbecker/food-101),
[Places-365](http://data.csail.mit.edu/places/places365/val_256.tar),
[Caltech-256](https://www.kaggle.com/jessicali9530/caltech256),
[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

For Food-101, remove hotdog class to avoid overlap.

## 2. Training
Currently, all code examples are assuming distributed launch with 4 multi GPUs.
To run the code with single GPU, remove `-m torch.distributed.launch --nproc_per_node=4`.

### Unlabeled one-class & multi-class 
To train unlabeled one-class & multi-class models in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset <DATASET> --model <NETWORK> --mode simclr_CSI --shift_trans_type rotation --batch_size 32 --one_class_idx <One-Class-Index>
```

> Option --one_class_idx denotes the in-distribution of one-class training.
> For multi-class training, set --one_class_idx as None.
> To run SimCLR simply change --mode to simclr.
> Total batch size should be 512 = 4 (GPU) * 32 (--batch_size option) * 4 (cardinality of shifted transformation set). 

### Labeled multi-class 
To train labeled multi-class model (confidence calibrated classifier) in the paper, run this command:

```train
# Representation train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset <DATASET> --model <NETWORK> --mode sup_simclr_CSI --shift_trans_type rotation --batch_size 32 --epoch 700
# Linear layer train
python train.py --mode sup_CSI_linear --dataset <DATASET> --model <NETWORK> --batch_size 32 --epoch 100 --shift_trans_type rotation --load_path <MODEL_PATH>
```

> To run SupCLR simply change --mode to sup_simclr, sup_linear for representation training and linear layer training respectively.
> Total batch size should be same as above. Currently only supports rotation for shifted transformation.

## 3. Evaluation

We provide the checkpoint of the CSI pre-trained model. Download the checkpoint from the following link:
- One-class CIFAR-10: [ResNet-18](https://drive.google.com/drive/folders/1z02i0G_lzrZe0NwpH-tnjpO8pYHV7mE9?usp=sharing)
- Unlabeled (multi-class) CIFAR-10: [ResNet-18](https://drive.google.com/file/d/1yUq6Si6hWaMa1uYyLDHk0A4BrPIa8ECV/view?usp=sharing)
- Unlabeled (multi-class) ImageNet-30: [ResNet-18](https://drive.google.com/file/d/1KucQWSik8RyoJgU-fz8XLmCWhvMOP7fT/view?usp=sharing)
- Labeled (multi-class) CIFAR-10: [ResNet-18](https://drive.google.com/file/d/1rW2-0MJEzPHLb_PAW-LvCivHt-TkDpRO/view?usp=sharing)

### Unlabeled one-class & multi-class
To evaluate my model on unlabeled one-class & multi-class out-of-distribution (OOD) detection setting, run this command:

```eval
python eval.py --mode ood_pre --dataset <DATASET> --model <NETWORK> --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx <One-Class-Index> --load_path <MODEL_PATH>
```

> Option --one_class_idx denotes the in-distribution of one-class evaluation.
> For multi-class evaluation, set --one_class_idx as None.
> The resize_factor & resize fix option fix the cropping size of RandomResizedCrop().
> For SimCLR evaluation, change --ood_score to simclr.

### Labeled multi-class 
To evaluate my model on labeled multi-class accuracy, ECE, OOD detection setting, run this command:

```eval
# OOD AUROC
python eval.py --mode ood --ood_score baseline_marginalized --print_score --dataset <DATASET> --model <NETWORK> --shift_trans_type rotation --load_path <MODEL_PATH>
# Accuray & ECE
python eval.py --mode test_marginalized_acc --dataset <DATASET> --model <NETWORK> --shift_trans_type rotation --load_path <MODEL_PATH>
```

> This option is for marginalized inference.
> For single inference (also used for SupCLR) change --ood_score baseline in first command,
> and --mode test_acc in second command.

## 4. Results

Our model achieves the following performance on:

### One-Class Out-of-Distribution Detection

| Method        | Dataset           |  AUROC (Mean) |
| --------------|------------------ | --------------|
| SimCLR        | CIFAR-10-OC       |      87.9%    |
| Rot+Trans     | CIFAR-10-OC       |      90.0%    |
| CSI (ours)    | CIFAR-10-OC       |      94.3%    |

We only show CIFAR-10 one-class result in this repo. For other setting, please see our paper.

### Unlabeled Multi-Class Out-of-Distribution Detection 

| Method        | Dataset           | OOD Dataset   | AUROC (Mean) |
| --------------|------------------ |---------------|--------------|
| Rot+Trans     | CIFAR-10          | CIFAR-100     |     82.5%    |
| CSI (ours)    | CIFAR-10          | CIFAR-100     |     89.3%    |

We only show CIFAR-10 to CIFAR-100 OOD detection result in this repo. For other OOD dataset results, see our paper.

### Labeled Multi-Class Result

| Method           | Dataset           | OOD Dataset   |  Acc  |  ECE  | AUROC (Mean) |
| ---------------- |------------------ |---------------|-------|-------|--------------|
| SupCLR           | CIFAR-10          | CIFAR-100     | 93.9% | 5.54% |     88.3%    |
| CSI (ours)       | CIFAR-10          | CIFAR-100     | 94.8% | 4.24% |     90.6%    |
| CSI-ensem (ours) | CIFAR-10          | CIFAR-100     | 96.0% | 3.64% |     92.3%    |

We only show CIFAR-10 with CIFAR-100 as OOD in this repo. For other dataset results, please see our paper.

## 5. New OOD dataset

<p align="center">
    <img src=figures/fixed_ood_benchmarks.png width="600"> 
</p>

We find that current benchmark datasets for OOD detection, are visually far from in-distribution datasets (e.g. CIFAR). 

To address this issue, we provide new datasets for OOD detection evaluation:
[LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing),
[ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing).
See the above figure for the visualization of current benchmark and our dataset.

To generate OOD datasets, run the following codes inside the `./datasets` folder:

```OOD dataset generation
# ImageNet FIX generation code
python imagenet_fix_preprocess.py 
# LSUN FIX generation code
python lsun_fix_preprocess.py
```

## Citation
```
@inproceedings{tack2020csi,
  title={CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances},
  author={Jihoon Tack and Sangwoo Mo and Jongheon Jeong and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
