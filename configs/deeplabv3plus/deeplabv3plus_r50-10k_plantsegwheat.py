_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/plantsegwheat.py',
    '../_base_/default_runtime.py', '../_base_/schedules/custom_30k.py'
]
crop_size = (256, 256)
train_dataloader = dict(batch_size=32)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=9),
    auxiliary_head=dict(num_classes=9))
