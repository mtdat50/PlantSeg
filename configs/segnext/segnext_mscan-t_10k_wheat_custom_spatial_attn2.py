_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCANWithCustomSpatialAttention',
        custom_version=2
    ),
)

# train_dataloader = dict(batch_size=2)
