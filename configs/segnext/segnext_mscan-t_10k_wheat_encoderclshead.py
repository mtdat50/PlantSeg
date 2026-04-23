# import torch


_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]


# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoderWithCls',
    cls_head=dict(
        type='ClsHead',
        in_channels=256,   # depends on backbone
        num_classes=9
    ),
    cls_loss_weight=0.3,
    decode_head=dict(
        loss_decode=[
           dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.7, avg_non_ignore=True)
        ],
        ignore_index=255
    ),
)

# dataset settings
train_dataloader = dict(batch_size=2)
