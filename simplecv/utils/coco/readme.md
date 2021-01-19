# COCO

## zip
```
unzip -l data_xxxx.zip
unzip data_xxxx.zip -d data_xxxx
zip -r data_xxxx.zip data_xxxx readme.md
for t in *.zip; do unzip -q "${t}" -d "zip_$(basename ${t} .zip)"; done
```

## dataset
```
import sys
SIMPLECV_PATH = "/workspace/SimpleCV"
!cd {SIMPLECV_PATH} && git log --oneline -1
if SIMPLECV_PATH not in sys.path:
    sys.path.insert(0, SIMPLECV_PATH)

from simplecv.utils.coco import abc
from simplecv.utils.coco import builder
from simplecv.utils.coco import selection

ann_dir = None
img_dir = "/workspace/notebooks/data_xxxx"
code_mapping = {
    "__BG": "__DEL",
    "__FG": "__XXX",
}

cvt_dir = abc.do_convert(img_dir, ann_dir, color=0)
coco_dir = builder.build_dataset(cvt_dir, code_mapping)
res = selection.split_dataset(coco_dir, seed=100, train_size=300, single_cls=True)
print("summary:\n", coco_dir, list(map(len, res)))

from simplecv.utils.coco import min_pos_iou
min_pos_iou.min_pos_iou(coco_dir + "/coco.json", crop_size=800, scale=8)

from simplecv.utils.coco import analyze_anchor
analyze_anchor.do_analyze(coco_dir + "/coco.json")
```
