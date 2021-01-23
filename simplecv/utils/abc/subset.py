import json
import shutil
from pathlib import Path


ANN_EXTENSIONS = set([".xml", ".json"])
SUPPORT_MODES = set(["csv", "pkl", "coco",  "json"])


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def _from_coco(root, json_file):
    coco = load_json(root / json_file)
    targets = [Path(img["file_name"]).stem for img in coco["images"]]

    keep_list = []
    targets = set(targets)
    for ann_file in sorted(root.glob("**/*")):
        if ann_file.suffix in ANN_EXTENSIONS and ann_file.stem in targets:
            keep_list.append(ann_file)
    return list(set(keep_list))


def _copyfile(src_file, out_dir):
    dst_file = out_dir / Path(src_file).name
    if not dst_file.exists():
        shutil.copyfile(src_file, dst_file)
    return str(dst_file)


def todo(root, mode="coco", out_dir=None, **kwargs):
    assert mode in SUPPORT_MODES

    if out_dir is None:
        out_dir = root.rstrip("/") + "_sub"
    shutil.rmtree(out_dir, ignore_errors=True)

    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "coco":
        keep_list = _from_coco(root, kwargs["json_file"])

    for file_path in keep_list:
        _copyfile(file_path, out_dir)
    return str(out_dir)
