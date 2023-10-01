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

save_step = 20
epoch = 0
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

    
    if epoch > 50:
        for param in model.parameters():
            param.requires_grad = True

    train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, train_exposure_loader=train_exposure_loader, logger=logger, **kwargs)

    model.eval()
    save_states = model.state_dict()
    save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)    
    print(P)
    if (epoch % save_step == 0) or 1==1:
        from evals.ood_pre import eval_ood_detection
        P.load_path = logger.logdir
        P.ood_layer = ("simclr", "shift")
        P.ood_score = ["CSI"]
        P.mode  = "ood_pre"
        P.OOD_sample = 10
        P.resize_factor = 0.54
        with torch.no_grad():
            auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                        train_loader=train_loader, simclr_aug=simclr_aug)
        P.mode  = "simclr_CSI"
        P.resize_factor = 0.08
        P.OOD_sample = 1
        P.ood_layer = "simclr"
        P.ood_score = ['norm_mean']


        
epoch += 1
if P.multi_gpu:
    save_states = model.module.state_dict()
else:
    save_states = model.state_dict()
save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

