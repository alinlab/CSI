from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['cifar100-versus-10', 'cifar10-versus-100', 'WBC', 'dtd', 'cifar10-corruption', 'Tomor_Detection', 'ucsd', 'mvtec-high-var', 'breastmnist', 'head-ct', 'fashion-mnist', 'mnist', 'cifar10', 'cifar100', 'imagenet', 'svhn-10', 'MVTecAD'],
                        default="cifar10", type=str)
    parser.add_argument('--normal_labels', help='normal_labels for high variation',
                        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13", type=str)
    parser.add_argument('--cifar_corruption_data', help='',
                        default="./CIFAR-10-C/defocus_blur.npy", type=str)                    
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--unfreeze_pretrain_model_epoch', help='unfreeze_pretrain_model',
                        default=50, type=int)
    parser.add_argument('--image_size', help='None: multi-class, Not None: one-class',
                        default=32, type=int)
    parser.add_argument('--save_step', help='None: multi-class, Not None: one-class',
                        default=20, type=int) 
    parser.add_argument('--fake_data_percent', help='',
                        default=0.0, type=float)
    parser.add_argument('--cutpast_data_percent', help='',
                        default=0.0, type=float)
    parser.add_argument('--main_count', help='count of normal data',
                        default=-1, type=int)
    parser.add_argument('--high_var', help='not used!',
                        default=0, choices=[0, 1], type=int)
    parser.add_argument('--model', help='Model',
                        choices=['pretrain-resnet152-corruption', 'pretrain-resnet152', 'vit_fitymi', 'vit', 'resnet18', 'resnet18_imagenet', 'pretrain-resnet18'], default="resnet18", type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='simclr', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
