_base_ = [
    'segnext_mscan-t_10k_rice.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCANWithCustomSpatialAttention2',
    ),
)

# train_dataloader = dict(batch_size=2)
