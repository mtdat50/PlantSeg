_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py',
    '../_base_/datasets/plantsegwheat.py',
    '../_base_/default_runtime.py',
    '../_base_/wheat_settings.py'
]
# crop_size = (256, 256)
# size = (512, 512)
size = (1024, 1024)
data_preprocessor = dict(size=size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    decode_head=dict(num_classes=9),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))


# Re-config the data sampler.
# train_dataloader = dict(batch_size=1)
