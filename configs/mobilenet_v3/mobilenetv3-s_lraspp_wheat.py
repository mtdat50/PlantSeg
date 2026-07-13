_base_ = [
    './mobilenetv3-l_lraspp_wheat.py',
]
# crop_size = (256, 256)
size = (512, 512)
# size = (1024, 1024)
data_preprocessor = dict(size=size)
norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    pretrained='open-mmlab://contrib/mobilenet_v3_small',
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(0, 1, 12),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 16, 576),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
