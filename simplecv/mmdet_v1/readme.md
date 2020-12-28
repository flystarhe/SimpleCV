# MMDet v1
Tested on [mmdetection v1](https://github.com/open-mmlab/mmdetection) commit `78529ec`.

## requirements
* mmcv
* pytorch
* torchvision

## environs
* Set environment variables `MMDET_PATH` and `SIMPLECV_PATH`
* Add `MMDET_PATH` and `SIMPLECV_PATH` to `PYTHONPATH`

## example
```
%matplotlib inline
import os
import time
from pathlib import Path

MMDET_PATH = '/data/sdv1/tmps/gits/mmdetection_78529ec'
SIMPLECV_PATH = '/data/sdv1/tmps/gits/SimpleCV'
!cd {SIMPLECV_PATH} && git log --oneline -1

os.environ['SIMPLECV_PATH'] = SIMPLECV_PATH
os.environ['MMDET_PATH'] = MMDET_PATH
os.chdir(SIMPLECV_PATH)
!pwd
```

Train:
```
DATA_ROOT = '/data/sdv1/tmps/data/coco'
DATA_TRAIN = 'data_train.json'
DATA_VAL = 'data_val.json'

os.environ['NUM_CLASSES'] = '10'

WORK_DIR = '/data/sdv1/tmps/results/{}_{}'.format(Path(DATA_ROOT).name, time.strftime('%m%d%H%M'))
CONFIG = 'simplecv/mmdet_v1/ext_configs/faster_rcnn_r50_fpn_1x.py'

!rm -rf data/coco && ln -s {DATA_ROOT} data/coco
!cd {DATA_ROOT} && rm -rf coco_train.json && ln -s {DATA_TRAIN} coco_train.json
!cd {DATA_ROOT} && rm -rf coco_val.json && ln -s {DATA_VAL} coco_val.json

TAIL = '{} --work_dir {} --launcher pytorch'.format(CONFIG, WORK_DIR)
!PYTHONPATH={SIMPLECV_PATH}:{MMDET_PATH} python -m torch.distributed.launch --nproc_per_node=4 simplecv/mmdet_v1/train_net.py {TAIL}
```

Test:
```
CHECKPOINT = '{}/epoch_{}.pth'.format(WORK_DIR, 12)

TAIL = '{} {} {} {} --gpus 2'.format(DATA_ROOT, 'none', CONFIG, CHECKPOINT)
!PYTHONPATH={SIMPLECV_PATH}:{MMDET_PATH} python simplecv/mmdet_v1/test_net.py {TAIL}
```
