_base_ = [
    'segnext_mscan-t_10k_plantsegwheat.py'
]
# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
auxhead0 = dict(
    type='LightHamHead',
    in_channels=[64],
    in_index=[1],
    channels=64,
    ham_channels=64,
    dropout_ratio=0.1,
    num_classes=9,
    norm_cfg=ham_norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
    ham_kwargs=dict(
        MD_S=1,
        MD_R=16,
        train_steps=6,
        eval_steps=7,
        inv_t=100,
        rand_init=True)
)
auxhead1 = auxhead0.copy()
auxhead1.update(
    in_channels=[64, 160],
    in_index=[1, 2],
    channels=160,
    ham_channels=160,
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3),
)
model = dict(
    auxiliary_head=[
        auxhead0,
        auxhead1
    ],
)

# dataset settings
# train_dataloader = dict(batch_size=2)
