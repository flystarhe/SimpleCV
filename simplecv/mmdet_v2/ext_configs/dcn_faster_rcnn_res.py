import os as __os
import os.path as __osp

__MMDET_PATH = __os.environ["MMDET_PATH"]
__CROP_SIZE = int(__os.environ["CROP_SIZE"])

_base_ = [
    __osp.join(__MMDET_PATH, 'configs/_base_/models/faster_rcnn_r50_fpn.py'),
    __osp.join(__MMDET_PATH, 'configs/_base_/datasets/coco_detection.py'),
    __osp.join(__MMDET_PATH, 'configs/_base_/schedules/schedule_2x.py'),
    __osp.join(__MMDET_PATH, 'configs/_base_/default_runtime.py'),
]

model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# image scale format: (w, h)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize2', test_mode=False, ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', height=__CROP_SIZE, width=__CROP_SIZE),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize2', test_mode=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_file='data/coco/annotations/train.json', img_prefix='data/coco/', pipeline=train_pipeline),
    test=dict(ann_file='data/coco/annotations/test.json', img_prefix='data/coco/', pipeline=test_pipeline),
    val=dict(ann_file='data/coco/annotations/val.json', img_prefix='data/coco/', pipeline=test_pipeline))
