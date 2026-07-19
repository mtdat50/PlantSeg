_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]

# spatial
model = dict(
    backbone=dict(
        type='MSCANWithCustomSpatialAttention',
        custom_version=17,
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
