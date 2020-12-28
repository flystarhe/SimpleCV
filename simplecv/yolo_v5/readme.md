# yolo

## environs
* Set environment variables `YOLO_PATH` and `SIMPLECV_PATH`
* Add `YOLO_PATH` and `SIMPLECV_PATH` to `PYTHONPATH`

## docker

* Python: 3.6
* PyTorch: 1.7.0
* YOLO PATH: `/usr/src`
* SSH INFO: `ssh root@ip -p 9001`
* Jupyter: `http://ip:9000/?token=hi`

```
docker pull flystarhe/simplecv:yolo5.3.1
docker run --gpus all -d -p 9000:9000 -p 9001:9001 --ipc=host --name yolo5 -v "$(pwd)":/workspace flystarhe/simplecv:yolo5.3.1
docker update --restart=always yolo5

docker save -o simplecv-yolo5.tar flystarhe/simplecv:yolo5.3.1
docker load -i simplecv-yolo5.tar
```

>`docker cp *.pth name:/root/.cache/torch/hub/checkpoints/`

`/workspace/(app_tornado.py + config.py + checkpoint.pth)`:
```python
import requests

url = "http://ip:9001/main"
vals = {"image": "/workspace/test.png"}

response = requests.get(url, params=vals)
print(response.status_code)
print(response.text)
```

## yolo
```
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

YOLO_PATH = "/usr/src/yolo"
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1

os.environ["SIMPLECV_PATH"] = SIMPLECV_PATH
os.environ["YOLO_PATH"] = YOLO_PATH
os.chdir(SIMPLECV_PATH)
!pwd

import time
EXPERIMENT_NAME = time.strftime("xxxx_%m%d_%H%M")
EXPERIMENT_NAME
```

Train:
```
?
```

## notes

* https://github.com/ultralytics/yolov5/wiki

```
ARG_DIST = "-m torch.distributed.launch --nproc_per_node 2"

python train.py  --batch-size 64 --data coco.yaml --weights yolov5l.pt --device 0
python ARG_DIST train.py --batch-size 64 --data coco.yaml --weights yolov5l.pt

python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
python test.py --weights yolov5x.pt --data coco.yaml --img 640 --iou 0.65
```
