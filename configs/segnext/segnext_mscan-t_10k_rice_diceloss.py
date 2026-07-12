# import torch


_base_ = [
    'segnext_mscan-t_10k_rice.py'
]


# model settings
model = dict(
    decode_head=dict(
        loss_decode=[
            # dict(type='DiceLoss', loss_weight=1),
            dict(type='DiceLoss', loss_weight=0.5),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)
        ],
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
