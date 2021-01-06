# mmdet v2
* Tested on [mmdetection v2.6.0](https://github.com/open-mmlab/mmdetection) commit `bd3306f`.
* Tested on [mmdetection v2.7.0](https://github.com/open-mmlab/mmdetection) commit `3e902c3`.

## environs
* Set environment variables `MMDET_PATH` and `SIMPLECV_PATH`
* Add `MMDET_PATH` and `SIMPLECV_PATH` to `PYTHONPATH`

## docker
* Python: 3.7
* PyTorch: 1.6.0
* MMDET: `/usr/src/mmdetection`
* [http://ip:9000/?token=hi](#) for `dev`
* `/usr/sbin/sshd -D -p 9000` for `ssh` mode
* `python /workspace/app_tornado.py 9000 ${@:2}` for `app` mode

```
docker pull registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:mmdet2.7
docker tag registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:mmdet2.7 simplecv:mmdet2.7

docker save -o mmdet2.7-20.12.tar simplecv:mmdet2.7
docker load -i mmdet2.7-20.12.tar

docker run --gpus device=0 -d -p 9000:9000 --ipc=host --name test -v "$(pwd)":/workspace simplecv:mmdet2.7 [dev|ssh|app] \
    /workspace/faster_rcnn_res1.py /workspace/epoch_48.pth 0.1 min
```

`python /workspace/app_tornado.py 9000 config.py checkpoint.pth [thr:float] [mode:str]`:
```python
import requests

url = "http://ip:9000/main"
vals = {"image": "/workspace/test.png"}

response = requests.get(url, params=vals)
print(response.status_code)
print(response.text)
```

## base
```
import os
import glob
import time

MMDET_PATH = "/usr/src/mmdetection"
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1

os.environ["MMDET_PATH"] = MMDET_PATH
os.environ["SIMPLECV_PATH"] = SIMPLECV_PATH

os.environ["MKL_THREADING_LAYER"] = "GNU"
EXPERIMENT_NAME = time.strftime("xxxx_%m%d_%H%M")
EXPERIMENT_NAME
```

## train
```
FLAG = "lr_1x_epochs_1x"
CONFIG_NAME = "faster_rcnn_res1"
DATA_ROOT = "/workspace/notebooks/data_xxxx_coco"
!mkdir -p data && rm -rf data/coco && ln -s {DATA_ROOT} data/coco

os.environ["CFG_OPTIONS"] = """
{
    "optimizer.lr":0.005,"total_epochs":12,
    "lr_config":dict(_delete_=True,policy="step",warmup="linear",warmup_iters=500,warmup_ratio=0.001,step=[8,11]),
    "evaluation.interval":12,"evaluation.metric":"bbox","log_config.interval":30,
    "data.train":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/train.json"),
    "data.test":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/test.json"),
    "data.val":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/val.json"),
    "data.samples_per_gpu":2,"data.workers_per_gpu":2,
    "model.neck.in_channels":[256,512,1024,2048],
    "model.neck.out_channels":256,
    "model.neck.start_level":0,
    "model.neck.num_outs":5,
    "model.neck.add_extra_convs":"on_output",
    "model.neck.relu_before_extra_convs":True,
    "model.rpn_head.anchor_generator.scales":[8],
    "model.rpn_head.anchor_generator.ratios":[0.5,1.0,2.0],
    "model.rpn_head.anchor_generator.strides":[4,8,16,32,64],
    "model.rpn_head.anchor_generator.base_sizes":[4,8,16,32,64],
    "model.roi_head.bbox_roi_extractor.featmap_strides":[4,8,16,32],
    "model.roi_head.bbox_head.num_classes":2,
    "model":dict(pretrained="torchvision://resnet50",backbone=dict(type="ResNet",depth=50,out_indices=(0,1,2,3),frozen_stages=1)),
    "train_cfg.rpn_proposal":dict(nms_across_levels=False,nms_pre=2000,nms_post=1000,max_num=1000,min_bbox_size=0),
    "test_cfg.rpn":dict(nms_across_levels=False,nms_pre=1000,nms_post=1000,max_num=1000,min_bbox_size=0),
    "test_cfg.rcnn":dict(score_thr=0.05,nms=dict(type="nms",iou_threshold=0.5),max_per_img=100),
}
"""

WORK_DIR = "/workspace/results/{}/{}".format(EXPERIMENT_NAME, FLAG)
CONFIG = "{}/simplecv/mmdet_v2/ext_configs/{}.py".format(SIMPLECV_PATH, CONFIG_NAME)

ARG_DIST = "-m torch.distributed.launch --nproc_per_node=2"
MY_SCRIPT = "{}/simplecv/mmdet_v2/py_train.py".format(SIMPLECV_PATH)
ARG_TRAIN = "{} --work-dir {} --launcher pytorch".format(CONFIG, WORK_DIR)

!PYTHONPATH={SIMPLECV_PATH}:{MMDET_PATH} python {ARG_DIST} {MY_SCRIPT} {ARG_TRAIN}
DEL_FILES = " ".join(glob.glob(WORK_DIR + "/epoch_*")[:-2])
!rm -rfv {DEL_FILES}
WORK_DIR
```

尝试不同主干网络：
```
"model":dict(pretrained="torchvision://resnet50",backbone=dict(type="ResNet",depth=50,out_indices=(0,1,2,3),frozen_stages=1)),
"model":dict(pretrained="torchvision://resnet101",backbone=dict(type="ResNet",depth=101,out_indices=(0,1,2,3),frozen_stages=1)),
"model":dict(pretrained="open-mmlab://resnext101_32x4d",backbone=dict(type="ResNeXt",depth=101,groups=32,base_width=4,num_stages=4,out_indices=(0,1,2,3),frozen_stages=1,norm_cfg=dict(type="BN",requires_grad=True),style="pytorch")),
"model":dict(pretrained="open-mmlab://resnext101_64x4d",backbone=dict(type="ResNeXt",depth=101,groups=64,base_width=4,num_stages=4,out_indices=(0,1,2,3),frozen_stages=1,norm_cfg=dict(type="BN",requires_grad=True),style="pytorch")),
# lr = 0.01 / 8 * batch_size, anchor scale range is [28, 448].
```

纵横比`ratios=h/s`异常时：
```
"train_cfg.rpn.assigner":dict(pos_iou_thr=0.7,neg_iou_thr=0.3,min_pos_iou=0.05,match_low_quality=True),
"train_cfg.rcnn.assigner":dict(pos_iou_thr=0.5,neg_iou_thr=0.5,min_pos_iou=0.05,match_low_quality=True),
# `max_iou_anchor_gt = 1 / (x + n -1)`, anchor shape `(x, x)`, gt shape `(1, nx)`, `x` in scales
# `if x=8; max_iou = 1 / (7 + n); ratios(8,16,24) <- 1/8,1/9,1/10`
# `if x=4; max_iou = 1 / (3 + n); ratios(4,8,12) <- 1/4,1/5,1/6`
```

尝试不同损失函数/权重：
```
"model.roi_head.bbox_head.loss_cls":dict(type="FocalLoss",use_sigmoid=False,loss_weight=1.0),
"model.roi_head.bbox_head.loss_bbox":dict(type="GIoULoss",loss_weight=1.0),
```

尝试不同学习策略：
```
"lr_config":dict(_delete_=True,policy="step",warmup="linear",warmup_iters=500,warmup_ratio=0.001,step=[8,11]),
"lr_config":dict(_delete_=True,policy="cyclic",by_epoch=False,target_ratio=(10,1e-4),cyclic_times=1,step_ratio_up=0.4),
"lr_config":dict(_delete_=True,policy="CosineRestart",periods=[8,4],restart_weights=[1.0,0.1],min_lr_ratio=1e-5),
"lr_config":dict(_delete_=True,policy="CosineAnnealing",min_lr_ratio=1e-5),
```

### VarifocalNet
```
FLAG = "lr_1x_epochs_1x"
CONFIG_NAME = "vfnet_r50_fpn"
DATA_ROOT = "/workspace/notebooks/data_xxxx_coco"
!mkdir -p data && rm -rf data/coco && ln -s {DATA_ROOT} data/coco

os.environ["CFG_OPTIONS"] = """
{
    "optimizer.lr":0.0025,"total_epochs":12,
    "lr_config":dict(_delete_=True,policy="step",warmup="linear",warmup_iters=500,warmup_ratio=0.001,step=[8,11]),
    "evaluation.interval":12,"evaluation.metric":"bbox","log_config.interval":30,
    "data.train":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/train.json"),
    "data.test":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/test.json"),
    "data.val":dict(img_prefix="data/coco/",ann_file="data/coco/annotations_100/val.json"),
    "data.samples_per_gpu":2,"data.workers_per_gpu":2,
    "model.neck.in_channels":[256,512,1024,2048],
    "model.neck.out_channels":256,
    "model.neck.start_level":1,
    "model.neck.num_outs":5,
    "model.neck.add_extra_convs":"on_output",
    "model.neck.relu_before_extra_convs":True,
    "model.bbox_head.strides":[8,16,32,64,128],
    "model.bbox_head.num_classes":2,
}
"""
```

>`lr = 0.01 / 16 * batch_size`, anchor scale range is `[14, ?]`.

### test
```
FLAG = "FLAG"
CONFIG_NAME = "CONFIG_NAME"
EXPERIMENT_NAME = "EXPERIMENT_NAME"
DATA_ROOT = "/workspace/notebooks/data_xxxx_coco"
!mkdir -p data && rm -rf data/coco && ln -s {DATA_ROOT} data/coco
os.environ["CFG_OPTIONS"] = "{}"

WORK_DIR = "/workspace/results/{}/{}".format(EXPERIMENT_NAME, FLAG)
CHECKPOINT = "{}/epoch_{}.pth".format(WORK_DIR, 24)
CONFIG = "{}/{}.py".format(WORK_DIR, CONFIG_NAME)

MY_SCRIPT = "{}/simplecv/mmdet_v2/py_test.py".format(SIMPLECV_PATH)
ARG_TEST = "{} {} {} {} --gpus 2".format(DATA_ROOT, "annotations_100/test.json", CONFIG, CHECKPOINT)

!PYTHONPATH={SIMPLECV_PATH}:{MMDET_PATH} python {MY_SCRIPT} {ARG_TEST}
WORK_DIR
```

## mmdet/tools
```
FLAG = "FLAG"
CONFIG_NAME = "CONFIG_NAME"
EXPERIMENT_NAME = "EXPERIMENT_NAME"
DATA_ROOT = "/workspace/notebooks/data_xxxx_coco"
!mkdir -p data && rm -rf data/coco && ln -s {DATA_ROOT} data/coco
MMDET_PATH = "/usr/src/mmdetection"

WORK_DIR = "/workspace/results/{}/{}".format(EXPERIMENT_NAME, FLAG)
CHECKPOINT = "{}/epoch_{}.pth".format(WORK_DIR, 24)
CONFIG = "{}/{}.py".format(WORK_DIR, CONFIG_NAME)

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

MY_SCRIPT = "{}/tools/test.py".format(MMDET_PATH)
!PYTHONPATH={MMDET_PATH} python {MY_SCRIPT} {CONFIG} {CHECKPOINT} --eval bbox

ARG_TEST = (" --out results.pkl"
            " --eval bbox"
            " --show-dir /workspace/results/xxxx-viz"
            " --show-score-thr 0.3"
            " --eval-options classwise=True")

MY_SCRIPT = "{}/tools/dist_test.sh".format(MMDET_PATH)
!PYTHONPATH={MMDET_PATH} /bin/bash {MY_SCRIPT} {CONFIG} {CHECKPOINT} 2 {ARG_TEST}
```

`tools/analyze_logs.py`:
```
FLAG = "*"
EXPERIMENT_NAME = "xxxx"
MMDET_PATH = "/usr/src/mmdetection"

import glob
from pathlib import Path
LOG_KEYS = ["loss_cls", "loss_bbox"]
LOG_FILES = glob.glob("/workspace/results/{}/{}/*.log.json".format(EXPERIMENT_NAME, FLAG))
LOG_LABELS = [Path(json_log).parent.name + ":" + key for key in LOG_KEYS for json_log in LOG_FILES]

MY_SCRIPT = "{}/tools/analyze_logs.py".format(MMDET_PATH)
OUT_FILE = "/workspace/results/{}/loss_{}.png".format(EXPERIMENT_NAME, FLAG)
LOG_FILES, LOG_KEYS, LOG_LABELS = " ".join(LOG_FILES), " ".join(LOG_KEYS), " ".join(LOG_LABELS)
!python {MY_SCRIPT} plot_curve {LOG_FILES} --keys {LOG_KEYS} --legend {LOG_LABELS} --out {OUT_FILE}
```

## notes
```
import sys
from pathlib import Path

SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1

if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.analyze import display_hardmini

pkl_file = "xxxx"
score_thr = {"*": 0.3}
output_dir = str(Path(pkl_file).parent) + "-viz"
kwargs = dict(show=False, clean_thr=0.3, clean_mode="min", match_mode="iou", pos_iou_thr=0.1, min_pos_iou=1e-3)
display_hardmini(pkl_file, score_thr, output_dir, simple=True, **kwargs)
```
