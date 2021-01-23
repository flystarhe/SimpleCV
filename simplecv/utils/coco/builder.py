import cv2 as cv
import json
import numpy as np
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEL_LABELS = set(["__DEL"])


def code_trans(data, key):
    if key in data:
        return data[key]
    elif "*" in data:
        return data["*"]
    return key


def bbox_norm(bbox, img_w, img_h):
    x, y, w, h = map(float, bbox)
    x, y = max(0, x), max(0, y)
    w = min(img_w - x, w)
    h = min(img_h - y, h)
    return [x, y, w, h]


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def parsing_dir(data_root):
    data_root = Path(data_root)
    ignore_file = data_root / "ignore.json"

    exclude = set()
    if ignore_file.exists():
        data = load_json(ignore_file)
        exclude = set([Path(file_name).stem for file_name in data])

    cache = defaultdict(dict)

    for img_path in sorted(data_root.glob("**/*.jpg")):
        file_name = img_path.relative_to(data_root).as_posix()
        cache[img_path.stem]["img"] = file_name

    for ann_path in sorted(data_root.glob("**/*.json")):
        file_name = ann_path.relative_to(data_root).as_posix()
        cache[ann_path.stem]["ann"] = file_name

    return [(v["img"], v["ann"]) for k, v in cache.items() if "img" in v and "ann" in v and k not in exclude]


def coco_labels(code_mapping, data_root, data):
    data_root = Path(data_root)

    labels = []
    for _, ann_path in data:
        ann_data = load_json(data_root / ann_path)
        labels.extend([shape["label"] for shape in ann_data["shapes"]])
    labels = set([code_trans(code_mapping, l) for l in labels])
    return sorted(labels - DEL_LABELS)


def copyfile(in_dir, out_dir, file_name):
    in_path = in_dir / file_name
    out_path = out_dir / "images" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        shutil.copyfile(in_path, out_path)
    return out_path.relative_to(out_dir).as_posix()


def fill_bbox(img_path, del_shapes):
    img_path = str(img_path)
    im = cv.imread(img_path, 1)

    for bbox in del_shapes:
        x, y, w, h = map(int, bbox)
        im[y: y + h, x: x + w] = 0

    cv.imwrite(img_path, im)


def build_dataset(in_dir, code_mapping):
    in_dir = Path(in_dir)
    out_dir = in_dir.name + "_coco"
    out_dir = in_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)

    data = parsing_dir(in_dir)
    labels = coco_labels(code_mapping, in_dir, data)
    print("[builder.pairs] {}".format(len(data)))

    imgs, anns = [], []
    img_id, ann_id = 0, 0
    for img_path, ann_path in data:
        ann_data = load_json(in_dir / ann_path)
        shapes = ann_data["shapes"]

        if len(shapes) == 0:
            print("non-shapes: " + img_path)

        img_w = ann_data["imageWidth"]
        img_h = ann_data["imageHeight"]
        img_path = copyfile(in_dir, out_dir, img_path)
        ann_path = copyfile(in_dir, out_dir, ann_path)

        img_id += 1
        img = dict(id=img_id,
                   width=img_w,
                   height=img_h,
                   file_name=img_path)
        imgs.append(img)

        del_shapes = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            label = code_trans(code_mapping, label)

            if shape_type == "rectangle":
                assert len(points) == 2, "[(x1, y1), (x2, y2)]"
                xy = np.array(points)
                x_min, y_min = np.min(xy, axis=0)
                x_max, y_max = np.max(xy, axis=0)
                w, h = x_max - x_min, y_max - y_min
                bbox = bbox_norm([x_min, y_min, w, h], img_w, img_h)
                x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
                points = [(x_mid, y_min), (x_max, y_mid), (x_mid, y_max), (x_min, y_mid)]
            elif shape_type == "polygon":
                assert len(points) >= 3, "[(x, y), (x, y), ...]"
                xy = np.array(points)
                x_min, y_min = np.min(xy, axis=0)
                x_max, y_max = np.max(xy, axis=0)
                w, h = x_max - x_min, y_max - y_min
                bbox = bbox_norm([x_min, y_min, w, h], img_w, img_h)
            else:
                raise NotImplementedError("Not Implemented shape_type={}".format(shape_type))

            if label in DEL_LABELS:
                del_shapes.append(bbox)
                continue

            ann_id += 1
            ann = dict(id=ann_id,
                       image_id=img_id,
                       category_id=labels.index(label),
                       segmentation=[np.asarray(points).flatten().tolist()],
                       area=np.prod(bbox[2:]),
                       bbox=bbox,
                       iscrowd=0)
            anns.append(ann)

        fill_bbox(out_dir / img_path, del_shapes)

    cats = [dict(id=i, name=name, supercategory="") for i, name in enumerate(labels)]
    coco = dict(images=imgs, annotations=anns, categories=cats)

    print("\n" + json.dumps(cats) + "\n")
    print("[builder.save] {}".format(out_dir))
    print("[builder.images] {}".format(len(imgs)))
    save_json(coco, out_dir / "coco.json")
    return str(out_dir)
