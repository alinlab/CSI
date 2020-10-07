import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

import models.transform_layers as TL
from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    if P.multi_gpu:
        rotation_linear = model.module.shift_cls_layer
        joint_linear = model.module.joint_distribution_layer
    else:
        rotation_linear = model.shift_cls_layer
        joint_linear = model.joint_distribution_layer

    if epoch == 1:
        # define optimizer and save in P (argument)
        milestones = [int(0.6 * P.epochs), int(0.75 * P.epochs), int(0.9 * P.epochs)]

        linear_optim = torch.optim.SGD(linear.parameters(),
                                       lr=1e-1, weight_decay=P.weight_decay)
        P.linear_optim = linear_optim
        P.linear_scheduler = lr_scheduler.MultiStepLR(P.linear_optim, gamma=0.1, milestones=milestones)

        rotation_linear_optim = torch.optim.SGD(rotation_linear.parameters(),
                                                 lr=1e-1, weight_decay=P.weight_decay)
        P.rotation_linear_optim = rotation_linear_optim
        P.rot_scheduler = lr_scheduler.MultiStepLR(P.rotation_linear_optim, gamma=0.1, milestones=milestones)

        joint_linear_optim = torch.optim.SGD(joint_linear.parameters(),
                                             lr=1e-1, weight_decay=P.weight_decay)
        P.joint_linear_optim = joint_linear_optim
        P.joint_scheduler = lr_scheduler.MultiStepLR(P.joint_linear_optim, gamma=0.1, milestones=milestones)

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['rot'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.eval()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images = hflip(images)  # 2B with hflip
        else:
            batch_size = images[0].size(0)
            images = images[0].to(device)

        labels = labels.to(device)
        images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])  # 4B
        rot_labels = torch.cat([torch.ones_like(labels) * k for k in range(4)], 0)  # B -> 4B
        joint_labels = torch.cat([labels + P.n_classes * i for i in range(4)], dim=0)

        images = simclr_aug(images)  # simclr augmentation
        _, outputs_aux = model(images, penultimate=True)
        penultimate = outputs_aux['penultimate'].detach()

        outputs = linear(penultimate[0:batch_size]) # only use 0 degree samples for linear eval
        outputs_rot = rotation_linear(penultimate)
        outputs_joint = joint_linear(penultimate)

        loss_ce = criterion(outputs, labels)
        loss_rot = criterion(outputs_rot, rot_labels)
        loss_joint = criterion(outputs_joint, joint_labels)

        ### CE loss ###
        P.linear_optim.zero_grad()
        loss_ce.backward()
        P.linear_optim.step()

        ### Rot loss ###
        P.rotation_linear_optim.zero_grad()
        loss_rot.backward()
        P.rotation_linear_optim.step()

        ### Joint loss ###
        P.joint_linear_optim.zero_grad()
        loss_joint.backward()
        P.joint_linear_optim.step()

        ### optimizer learning rate ###
        lr = P.linear_optim.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_ce.item(), batch_size)
        losses['rot'].update(loss_rot.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossR %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['rot'].value))
        check = time.time()

    P.linear_scheduler.step()
    P.rot_scheduler.step()
    P.joint_scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossR %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['rot'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_rot', losses['rot'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
