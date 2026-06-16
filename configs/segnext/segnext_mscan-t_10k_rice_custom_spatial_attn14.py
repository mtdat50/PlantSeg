_base_ = [
    'segnext_mscan-t_10k_rice.py'
]
# model settings
embed_dims=[32, 64, 160, 256]
model = dict(
    backbone=dict(
        type='MSCANWithCustomSpatialAttention',
        embed_dims=embed_dims,
        hidden_embed_dims=[int(x * 2) for x in embed_dims],
        mlp_ratios=[6, 6, 4, 4],
        custom_version=8
    ),
)

# train_dataloader = dict(batch_size=2)
