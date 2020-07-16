import torch
import torch.nn as nn
import torch.nn.functional as F


def update_learning_rate(P, optimizer, cur_epoch, n, n_total):

    cur_epoch = cur_epoch - 1

    lr = P.lr_init
    if P.optimizer == 'sgd' or 'lars':
        DECAY_RATIO = 0.1
    elif P.optimizer == 'adam':
        DECAY_RATIO = 0.3
    else:
        raise NotImplementedError()

    if P.warmup > 0:
        cur_iter = cur_epoch * n_total + n
        if cur_iter <= P.warmup:
            lr *= cur_iter / float(P.warmup)

    if cur_epoch >= 0.5 * P.epochs:
        lr *= DECAY_RATIO
    if cur_epoch >= 0.75 * P.epochs:
        lr *= DECAY_RATIO
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)


def cross_entropy_soft(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), F.softmax(targets, dim=1),
                    reduction=reduction)


def target_nll_loss(inputs, targets, reduction='none'):
    inputs_t = -F.nll_loss(inputs, targets, reduction='none')
    logit_diff = inputs - inputs_t.view(-1, 1)
    logit_diff = logit_diff.scatter(1, targets.view(-1, 1), -1e8)
    diff_max = logit_diff.max(1)[0]

    if reduction == 'sum':
        return diff_max.sum()
    elif reduction == 'mean':
        return diff_max.mean()
    elif reduction == 'none':
        return diff_max
    else:
        raise NotImplementedError()


def target_nll_c(inputs, targets, reduction='none'):
    conf = torch.softmax(inputs, dim=1)
    conf_t = -F.nll_loss(conf, targets, reduction='none')
    conf_diff = conf - conf_t.view(-1, 1)
    conf_diff = conf_diff.scatter(1, targets.view(-1, 1), -1)
    diff_max = conf_diff.max(1)[0]

    if reduction == 'sum':
        return diff_max.sum()
    elif reduction == 'mean':
        return diff_max.mean()
    elif reduction == 'none':
        return diff_max
    else:
        raise NotImplementedError()