_base_ = [
    '../_base_/datasets/rice.py',
    '../_base_/default_runtime.py',
    '../_base_/rice_settings.py'
]

crop_size = (256, 256)

train_dataloader = dict(batch_size=16)
# train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

norm_cfg = dict(
    type='SyncBN',
    eps=0.001,
    requires_grad=True)

checkpoint_file='open-mmlab://contrib/mobilenet_v3_small'
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        out_indices=(1, 12),
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=576,
        in_index=1,
        channels=256,
        dilations=(1, 12, 24, 36),
        c1_in_channels=16,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(
        mode='whole'))
