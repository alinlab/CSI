import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True, local_rank=0):
        self.local_rank = local_rank
        if self.local_rank == 0:
            if not os.path.exists("./logs/"):
                os.mkdir("./logs/")

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        if self.local_rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.local_rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank == 0:
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank == 0:
            self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank == 0:
            self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last'):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, 'last.model')
    last_optim = os.path.join(logdir, 'last.optim')
    last_config = os.path.join(logdir, 'last.config')

    opt = {
        'epoch': epoch,
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_linear_checkpoint(logdir, mode='last'):
    if mode == 'last':
        linear_optim_path = os.path.join(logdir, 'last.linear_optim')
    elif mode == 'best':
        linear_optim_path = os.path.join(logdir, 'best.linear_optim')
    else:
        raise NotImplementedError()

    print("=> Loading linear optimizer checkpoint from '{}'".format(logdir))
    if os.path.exists(linear_optim_path):
        linear_optim_state = torch.load(linear_optim_path)
        return linear_optim_state
    else:
        return None


def save_linear_checkpoint(linear_optim_state, logdir):
    last_linear_optim = os.path.join(logdir, 'last.linear_optim')
    torch.save(linear_optim_state, last_linear_optim)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def make_model_diagrams(probs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    confidences, predictions = probs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 2, figsize=(4, 2.5))

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in
                   zip(bins[:-1], bins[1:])]
    bin_corrects = [torch.mean(accuracies[bin_index]) for bin_index in bin_indices]
    bin_scores = [torch.mean(confidences[bin_index]) for bin_index in bin_indices]

    confs = rel_ax.bar(bins[:-1], bin_corrects.numpy(), width=width)
    gaps = rel_ax.bar(bins[:-1], (bin_scores - bin_corrects).numpy(), bottom=bin_corrects.numpy(), color=[1, 0.7, 0.7],
                      alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

    # Clean up
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()
    return f


