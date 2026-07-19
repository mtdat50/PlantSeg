_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]

# default spatial attn
crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
cls_loss_weight = 0.1
model = dict(
    type='EncoderDecoderWithCls',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MSCANWithChannelAttention',
        channel_attention='ECA',
    ),
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
custom_hooks = [
    dict(
        type='WarmupHook',
        warmup_iters=0
    )
]

# dataset settings
# train_dataloader = dict(batch_size=2)
