_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
model = dict(
    backbone=dict(
        type='MSCANWithChannelAttention',
        channel_attn='SE'
    ),
)
custom_hooks = [
    dict(
        type='WarmupHook',
        warmup_iters=5000
    )
]

# train_dataloader = dict(batch_size=2)
