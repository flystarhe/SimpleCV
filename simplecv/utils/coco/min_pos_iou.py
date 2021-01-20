import json
import numpy as np


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def max_iou(w, h, anchors):
    data = []
    for anchor in anchors:
        I = min(w, anchor) * min(h, anchor)
        U = (w * h) + (anchor ** 2) - I
        data.append(I / (U + 1e-5))
    return max(data)


def min_pos_iou(coco_file, crop_size=800, scale=8):
    base_sizes = [4, 8, 16, 32, 64]
    anchors = [scale * x for x in base_sizes]

    coco = load_json(coco_file)

    data = []
    for ann in coco["annotations"]:
        w, h = ann["bbox"][2:]
        w = min(crop_size, w)
        h = min(crop_size, h)
        data.append(max_iou(w, h, anchors))

    q = [i / 10 for i in range(0, 10)]
    res = np.quantile(data, q, interpolation="higher")

    print(["{:.2f}:{:.2f}".format(a, b) for a, b in zip(q, res)])

    return q, res
