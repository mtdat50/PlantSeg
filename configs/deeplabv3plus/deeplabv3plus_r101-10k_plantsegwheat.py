_base_ = './deeplabv3plus_r50-10k_plantsegwheat.py'
train_dataloader = dict(batch_size=16)
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
