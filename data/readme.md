# ReadMe

## Notes
```
!cd {HOME}
!mkdir -p data
!rm -rf data/coco
!ln -s {DATA_ROOT} data/coco
```

options:
```
!cd {DATA_ROOT} && rm -rf train2017 && ln -s images train2017
!cd {DATA_ROOT} && rm -rf val2017 && ln -s images val2017
```

coco detection config:
```
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))
```
