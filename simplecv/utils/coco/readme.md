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

img_dir = "/workspace/notebooks/data_xxxx"
ann_dir = "/workspace/notebooks/data_xxxx"
ext_file = None
# coco format json file path or csv file path
code_mapping = {
    "__BG": "__DEL",
}

cvt_dir = abc.do_convert(img_dir, ann_dir, ext_file, suffix=".jpg", color=1)
coco_dir = builder.build_dataset(cvt_dir, code_mapping)

res = selection.split_dataset(coco_dir, seed=100, train_size=300, single_cls=True)
print("summary:\n", coco_dir, list(map(len, res)))

from simplecv.utils.coco import analyze_bbox
analyze_bbox.bbox_quantile(coco_dir + "/coco.json", crop_size=640, scales=[8], ratios=[0.5, 1.0, 2.0])

from simplecv.utils.coco import analyze_anchor
analyze_anchor.do_analyze(coco_dir + "/coco.json")
```
