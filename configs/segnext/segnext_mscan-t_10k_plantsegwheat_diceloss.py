_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]


model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', loss_weight=1),
        ],
    ),
)

# dataset settings
# train_dataloader = dict(batch_size=2)
