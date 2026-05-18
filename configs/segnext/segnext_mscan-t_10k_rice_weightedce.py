# import torch


_base_ = [
    'segnext_mscan-t_10k_rice.py'
]


# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=[
            # dict(type='DiceLoss', ignore_index=0, loss_weight=1),
           # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[0.5, 2, 1, 1, 1, 1, 1, 1, 1], avg_non_ignore=True)
           dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[1.09, 8.11, 8.9, 6.77, 12.36, 4.85, 6.69, 8.12, 7.1], avg_non_ignore=True)
           # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[0.17, 4.19, 4.37, 3.83, 5.03, 3.16, 3.8, 4.19, 3.92], avg_non_ignore=True)
        ],
        ignore_index=255
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
