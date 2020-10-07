def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}'

    if mode == 'sup_linear':
        from .sup_linear import train
    elif mode == 'sup_CSI_linear':
        from .sup_CSI_linear import train
    elif mode == 'sup_simclr':
        from .sup_simclr import train
    elif mode == 'sup_simclr_CSI':
        assert P.batch_size == 32
        # currently only support rotation
        from .sup_simclr_CSI import train
    else:
        raise NotImplementedError()

    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname


def update_comp_loss(loss_dict, loss_in, loss_out, loss_diff, batch_size):
    loss_dict['pos'].update(loss_in, batch_size)
    loss_dict['neg'].update(loss_out, batch_size)
    loss_dict['diff'].update(loss_diff, batch_size)


def summary_comp_loss(logger, tag, loss_dict, epoch):
    logger.scalar_summary(f'{tag}/pos', loss_dict['pos'].average, epoch)
    logger.scalar_summary(f'{tag}/neg', loss_dict['neg'].average, epoch)
    logger.scalar_summary(f'{tag}', loss_dict['diff'].average, epoch)

