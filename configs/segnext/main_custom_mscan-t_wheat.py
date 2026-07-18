_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]


checkpoint_file = '~/.cache/torch/hub/checkpoints/mscan_t_20230227-119e8c9f.pth'  # noqa
# all
# cls_loss_weight = 0.1
# # model settings
# model = dict(
#     type='EncoderDecoderWithCls',
#     backbone=dict(
#         type='MainCustomMSCAN',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         custom_version=17,
#         channel_attention='ECA',
#     ),
#     cls_head=dict(
#         type='ClsHead',
#         in_channels=256,
#         num_classes=9
#     ),
#     cls_loss_weight=cls_loss_weight,
#     cls_decay_iters=10000,
#     decode_head=dict(
#         loss_decode=[
#             dict(type='DiceLoss', loss_weight=(1 - cls_loss_weight) / 2),
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight) / 2)
#         ],
#         ignore_index=255
#     ),
# )
#=========================================
#no channel attn
# cls_loss_weight = 0.1
# model = dict(
#     type='EncoderDecoderWithCls',
#     backbone=dict(
#         type='MainCustomMSCAN',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         custom_version=17,
#     ),
#     cls_head=dict(
#         type='ClsHead',
#         in_channels=256,
#         num_classes=9
#     ),
#     cls_loss_weight=cls_loss_weight,
#     cls_decay_iters=10000,
#     decode_head=dict(
#         loss_decode=[
#             dict(type='DiceLoss', loss_weight=(1 - cls_loss_weight) / 2),
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight) / 2)
#         ],
#         ignore_index=255
#     ),
# )

# #=========================================
# # default spatial attn
# cls_loss_weight = 0.1
# model = dict(
#     type='EncoderDecoderWithCls',
#     backbone=dict(
#         type='MSCANWithChannelAttention',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         custom_version=17,
#         channel_attention='ECA',
#     ),
#     cls_head=dict(
#         type='ClsHead',
#         in_channels=256,
#         num_classes=9
#     ),
#     cls_loss_weight=cls_loss_weight,
#     cls_decay_iters=10000,
#     decode_head=dict(
#         loss_decode=[
#             dict(type='DiceLoss', loss_weight=(1 - cls_loss_weight) / 2),
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight) / 2)
#         ],
#         ignore_index=255
#     ),
# )
#
#=========================================
#without diceloss
# cls_loss_weight = 0.1
# model = dict(
#     type='EncoderDecoderWithCls',
#     backbone=dict(
#         type='MainCustomMSCAN',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         custom_version=17,
#         channel_attention='ECA',
#     ),
#     cls_head=dict(
#         type='ClsHead',
#         in_channels=256,
#         num_classes=9
#     ),
#     cls_loss_weight=cls_loss_weight,
#     cls_decay_iters=10000,
#     decode_head=dict(
#         loss_decode=[
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight))
#         ],
#         ignore_index=255
#     ),
# )
#
#====================================
# no cls
# cls_loss_weight = 0.
# model = dict(
#     type='EncoderDecoder',
#     backbone=dict(
#         type='MainCustomMSCAN',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         custom_version=17,
#         channel_attention='ECA',
#     ),
#     decode_head=dict(
#         loss_decode=[
#             dict(type='DiceLoss', loss_weight=(1 - cls_loss_weight) / 2),
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=(1 - cls_loss_weight) / 2)
#         ],
#         ignore_index=255
#     ),
# )
#
#====================================
# spatial + channel
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MainCustomMSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True),
        custom_version=17,
        channel_attention='ECA',
    ),
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1)
        ],
        ignore_index=255
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
