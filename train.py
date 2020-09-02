from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier

if 'sup' in P.mode:
    from training.sup import setup
else:
    from training.unsup import setup
train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {}
    kwargs['linear'] = linear
    kwargs['linear_optim'] = linear_optim
    kwargs['simclr_aug'] = simclr_aug

    train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)

    model.eval()

    if epoch % P.save_step == 0 and P.local_rank == 0:
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

    if epoch % P.error_step == 0 and ('sup' in P.mode):
        error = test_classifier(P, model, test_loader, epoch, logger=logger)

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_error', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))
