# ABC

## patch
```
import sys
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1
if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.abc import patch
img_dir = "xxx"
out_dir = "xxx"
patch.do_patch(img_dir, out_dir, patch_size=800, color_mode=1)
```

## show
```
import sys
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1
if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.abc import show
img_dir = "xxx"
out_dir = "xxx"
show.show_dataset(img_dir, out_dir)
```

## subset
```
import sys
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1
if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.abc import subset
src_dir = "xxx"
dst_dir = "xxx"
subset.todo(src_dir, "coco", dst_dir, json_file="annotations_100/train.json")
```
