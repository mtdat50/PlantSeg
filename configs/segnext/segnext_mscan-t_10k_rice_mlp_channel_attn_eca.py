_base_ = [
    'segnext_mscan-t_10k_rice.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCAN',
        mlp_channel_attention_type='ECA'
    ),
)

# train_dataloader = dict(batch_size=2)
