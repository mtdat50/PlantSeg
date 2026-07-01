_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]


model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', loss_weight=0.5),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)
        ],
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
