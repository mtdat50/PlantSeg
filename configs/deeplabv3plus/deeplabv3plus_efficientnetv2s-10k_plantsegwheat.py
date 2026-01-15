_base_ = [
    '../_base_/datasets/plantsegwheat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/custom_30k.py'
]
train_dataloader = dict(batch_size=16)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TIMMBackbone',
        model_name='tf_efficientnetv2_s',
        features_only=True,
        pretrained=True,
        in_channels=3,
        out_indices=(1, 2, 3, 4),  # Use multiple feature levels
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=256,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=160,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
