# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mmengine.utils import mkdir_or_exist, progressbar
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='OrRd'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=300)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    colorbar = plt.colorbar(mappable=im, ax=ax)
    colorbar.ax.tick_params(labelsize=20)  # 设置 colorbar 标签的字体大小

    title_font = {'weight': 'bold', 'size': 20}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 40}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='k',
                size=20)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        mkdir_or_exist(save_dir)
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / (len(values) - 1))**0.5
    return mean, std


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume


    mious = []
    mdices = []
    mprecisions = []
    mrecalls = []

    # prepare confusion_matrix storage on disk
    os.remove("confusion_matrix.npy") if os.path.exists("confusion_matrix.npy") else None

    # start training
    for i in range(5):
        train_seed = 2**i
        cfg.randomness = dict(seed=train_seed, deterministic=False, diff_rank_seed=True)
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        runner.train()
        runner.save_checkpoint(
            out_dir=cfg.work_dir,
            filename=f'model_{i}.pth', 
            save_optimizer=True,       # Include optimizer state for resuming
            save_param_scheduler=True, # Include scheduler state
            meta={'comment': 'Manual save'} # Optional metadata
        )

        test_seeds = [2**j for j in range(10)]
        for test_seed in test_seeds:
            runner.cfg.randomness = dict(seed=test_seed, deterministic=False, diff_rank_seed=True)

            metrics = runner.test()
            mious.append(metrics['mIoU'])
            mdices.append(metrics['mDice'])
            mprecisions.append(metrics['mPrecision'])
            mrecalls.append(metrics['mRecall'])

    pd.DataFrame({
        'mIoU': mious,
        'mDice': mdices,
        'mPrecision': mprecisions,
        'mRecall': mrecalls
    }).to_csv(osp.join(cfg.work_dir, 'multi_train_test_results.csv'), index=False)

    mean_miou, std_miou = mean_std(mious)
    mean_mdice, std_mdice = mean_std(mdices)
    mean_mprecision, std_mprecision = mean_std(mprecisions)
    mean_mrecall, std_mrecall = mean_std(mrecalls)

    print(
        f'mIoU: {mean_miou:.2f} ± {std_miou:.2f}'
        f'mDice: {mean_mdice:.2f} ± {std_mdice:.2f}'
        f'mPrecision: {mean_mprecision:.2f} ± {std_mprecision:.2f}'
        f'mRecall: {mean_mrecall:.2f} ± {std_mrecall:.2f}'
    )

    confusion_matrix = np.load("confusion_matrix.npy")
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    plot_confusion_matrix(
        confusion_matrix,
        dataset.METAINFO['classes'],
        save_dir=cfg.work_dir,
        show=False,
        title="",
        color_theme="winter")

    os.rename("confusion_matrix.npy", os.path.join(cfg.work_dir, "confusion_matrix.npy"))

if __name__ == '__main__':
    main()
