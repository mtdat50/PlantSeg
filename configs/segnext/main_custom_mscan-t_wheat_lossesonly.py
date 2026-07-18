_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]

checkpoint_file = '~/.cache/torch/hub/checkpoints/mscan_t_20230227-119e8c9f.pth'  # noqa
cls_loss_weight = 0.1
# model settings
model = dict(
    type='EncoderDecoderWithCls',
    cls_head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=9
    ),
    cls_loss_weight=cls_loss_weight,
    cls_decay_iters=10000,
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', loss_weight=(1 - cls_loss_weight) / 2),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight) / 2)
        ],
        ignore_index=255
    ),
)
#
# dataset settings
# train_dataloader = dict(batch_size=2)
