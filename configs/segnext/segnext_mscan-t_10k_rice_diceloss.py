# import torch


_base_ = [
    'segnext_mscan-t_10k_rice.py'
]


# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', loss_weight=1),
           # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[0.1, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], avg_non_ignore=True)
        ],
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
