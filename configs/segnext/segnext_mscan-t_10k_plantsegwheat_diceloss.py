_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', ignore_index=0, loss_weight=1),
            # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=.25)
        ]
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=16)
