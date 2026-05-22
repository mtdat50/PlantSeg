# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import pandas as pd

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


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

    # start training
    train_seeds = [2**i for i in range(5)]
    for train_seed in train_seeds:
        cfg.randomness = dict(seed=train_seed, deterministic=False, diff_rank_seed=True)
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        runner.train()

        test_seeds = [2**i for i in range(10)]
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



if __name__ == '__main__':
    main()
