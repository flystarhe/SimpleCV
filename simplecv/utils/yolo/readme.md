# yolo

## zip
```
unzip -l data_xxxx.zip
unzip data_xxxx.zip -d data_xxxx
zip -r data_xxxx.zip data_xxxx readme.md
```

## dataset
```
import sys
SIMPLECV_PATH = ""
!cd {SIMPLECV_PATH} && git log --oneline -1
if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.yolo import run2 as run

json_dir = "annotations"
coco_dir = "data_xxxx_coco"
run.convert_coco_json(coco_dir, json_dir)
```
