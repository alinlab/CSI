import time

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, Supervised_NT_xent
from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    # currently only support rotation shifting augmentation
    assert simclr_aug is not None
    assert P.sim_lambda == 1.0

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)

        images1 = torch.cat([torch.rot90(images1, rot, (2, 3)) for rot in range(4)])  # 4B
        images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])  # 4B
        images_pair = torch.cat([images1, images2], dim=0)  # 8B

        labels = labels.to(device)
        rot_sim_labels = torch.cat([labels + P.n_classes * i for i in range(4)], dim=0)
        rot_sim_labels = rot_sim_labels.to(device)

        images_pair = simclr_aug(images_pair)  # simclr augment
        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss_sim = Supervised_NT_xent(sim_matrix, labels=rot_sim_labels,
                                      temperature=0.07, multi_gpu=P.multi_gpu) * P.sim_lambda

        ### total loss ###
        loss = loss_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][4 * batch_size: 5 * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        ### Log losses ###
        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value))

        check = time.time()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

