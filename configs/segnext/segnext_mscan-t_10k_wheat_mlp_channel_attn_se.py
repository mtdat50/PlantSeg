_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCAN',
        mlp_channel_attention_type='SE'
    ),
)

# train_dataloader = dict(batch_size=2)
