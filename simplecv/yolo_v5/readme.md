# yolo

## environs
* Set environment variables `YOLO_PATH` and `SIMPLECV_PATH`
* Add `YOLO_PATH` and `SIMPLECV_PATH` to `PYTHONPATH`

## docker

* Python: 3.6
* PyTorch: 1.7.0
* YOLO PATH: `/usr/src`
* [http://ip:9000/?token=hi](#) for `dev`
* `/usr/sbin/sshd -D -p 9000` for `ssh` mode
* `python /workspace/app_tornado.py 9000 ${@:2}` for `app` mode

```
docker pull registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:yolo5.3.1
docker tag registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:yolo5.3.1 simplecv:yolo5.3.1

docker save -o yolo5.3.1-20.12.tar simplecv:yolo5.3.1
docker load -i yolo5.3.1-20.12.tar

docker run --gpus device=0 -d -p 9000:9000 --ipc=host --name test -v "$(pwd)":/workspace simplecv:yolo5.3.1 [dev|ssh|app]
docker update --restart=always test
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
