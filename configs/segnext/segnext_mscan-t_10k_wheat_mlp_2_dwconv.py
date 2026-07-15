_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCAN',
        mlp_ratios=[3, 3, 2, 2],
        n_dwconv=2
    ),
)

# train_dataloader = dict(batch_size=2)
