import time
import itertools

import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import models.transform_layers as TL
from utils.temperature_scaling import _ECELoss
from utils.utils import AverageMeter, set_random_seed, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ece_criterion = _ECELoss().to(device)


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


def test_classifier(P, model, loader, steps, marginal=False, logger=None):
    error_top1 = AverageMeter()
    error_calibration = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)

        if marginal:
            outputs = 0
            for i in range(4):
                rot_images = torch.rot90(images, i, (2, 3))
                _, outputs_aux = model(rot_images, joint=True)
                outputs += outputs_aux['joint'][:, P.n_classes * i: P.n_classes * (i + 1)] / 4.
        else:
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        ece = ece_criterion(outputs, labels) * 100
        error_calibration.update(ece.item(), batch_size)

        if n % 100 == 0:
            log_('[Test %3d] [Test@1 %.3f] [ECE %.3f]' %
                 (n, error_top1.value, error_calibration.value))

    log_(' * [Error@1 %.3f] [ECE %.3f]' %
         (error_top1.average, error_calibration.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)
        logger.scalar_summary('eval/ece', error_calibration.average, steps)

    model.train(mode)

    return error_top1.average


def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    for ood_score in ood_scores:
        # compute scores for ID and OOD samples
        score_func = get_ood_score_func(P, model, ood_score, simclr_aug=simclr_aug)

        save_path = f'plot/score_in_{P.dataset}_{ood_score}'
        if P.one_class_idx is not None:
            save_path += f'_{P.one_class_idx}'

        scores_id = get_scores(id_loader, score_func)

        if P.save_score:
            np.save(f'{save_path}.npy', scores_id)

        for ood, ood_loader in ood_loaders.items():
            if ood == 'interp':
                scores_ood = get_scores_interp(id_loader, score_func)
                auroc_dict['interp'][ood_score] = get_auroc(scores_id, scores_ood)
            else:
                scores_ood = get_scores(ood_loader, score_func)
                auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood)

            if P.save_score:
                np.save(f'{save_path}_out_{ood}.npy', scores_ood)

    return auroc_dict


def get_ood_score_func(P, model, ood_score, simclr_aug=None):
    def score_func(x):
        return compute_ood_score(P, model, ood_score, x, simclr_aug=simclr_aug)
    return score_func


def get_scores(loader, score_func):
    scores = []
    for i, (x, _) in enumerate(loader):
        s = score_func(x.to(device))
        assert s.dim() == 1 and s.size(0) == x.size(0)

        scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores)


def get_scores_interp(loader, score_func):
    scores = []
    for i, (x, _) in enumerate(loader):
        x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
        last = x  # save the last batch
        s = score_func(x_interp.to(device))
        assert s.dim() == 1 and s.size(0) == x.size(0)

        scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores)


def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)


def compute_ood_score(P, model, ood_score, x, simclr_aug=None):
    model.eval()

    if ood_score == 'clean_norm':
        _, output_aux = model(x, penultimate=True, simclr=True)
        score = output_aux[P.ood_layer].norm(dim=1)
        return score

    elif ood_score == 'similar':
        assert simclr_aug is not None  # require custom simclr augmentation
        sample_num = 2  # fast evaluation
        feats = get_features(model, simclr_aug, x, layer=P.ood_layer, sample_num=sample_num)
        feats_avg = sum(feats) / len(feats)

        scores = []
        for seed in range(sample_num):
            sim = torch.cosine_similarity(feats[seed], feats_avg)
            scores.append(sim)
        return sum(scores) / len(scores)

    elif ood_score == 'baseline':
        outputs, outputs_aux = model(x, penultimate=True)
        scores = F.softmax(outputs, dim=1).max(dim=1)[0]
        return scores

    elif ood_score == 'baseline_marginalized':

        total_outputs = 0
        for i in range(4):
            x_rot = torch.rot90(x, i, (2, 3))
            outputs, outputs_aux = model(x_rot, penultimate=True, joint=True)
            total_outputs += outputs_aux['joint'][:, P.n_classes * i:P.n_classes * (i + 1)]

        scores = F.softmax(total_outputs / 4., dim=1).max(dim=1)[0]
        return scores

    else:
        raise NotImplementedError()


def get_features(model, simclr_aug, x, layer='simclr', sample_num=1):
    model.eval()

    feats = []
    for seed in range(sample_num):
        set_random_seed(seed)
        x_t = simclr_aug(x)
        with torch.no_grad():
            _, output_aux = model(x_t, penultimate=True, simclr=True, shift=True)
        feats.append(output_aux[layer])
    return feats
