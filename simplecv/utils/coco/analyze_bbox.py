import json
import numpy as np


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def best_iou(w, h, anchors, ratios):
    data = []
    for anchor in anchors:
        for ratio in ratios:
            c = np.sqrt(ratio)
            anchor_w = anchor / c
            anchor_h = anchor * c
            I = min(w, anchor_w) * min(h, anchor_h)
            U = (w * h) + (anchor ** 2) - I
            data.append(I / U)
    return max(data)


def bbox_quantile(coco_file, crop_size=640, scales=[8], ratios=[0.5, 1.0, 2.0]):
    ious = []
    areas = []
    h_ratios = []
    min_sizes = []
    coco = load_json(coco_file)
    base_sizes = [4, 8, 16, 32, 64]
    anchors = [s * x for s in scales for x in base_sizes]
    for ann in coco["annotations"]:
        w, h = ann["bbox"][2:]
        areas.append(w * h)
        h_ratios.append(h / w)
        min_sizes.append(min(w, h))
        w, h = min(crop_size, w), min(crop_size, h)
        ious.append(best_iou(w, h, anchors, ratios))

    q = [i / 20 for i in range(0, 21)]

    res = np.quantile(ious, q, interpolation="higher")
    print("ious:", ["{:.2f}:{:.2f}".format(a, b) for a, b in zip(q, res)])

    res = np.quantile(areas, q, interpolation="higher")
    print("areas:", ["{:.2f}:{:.2f}".format(a, b) for a, b in zip(q, res)])

    res = np.quantile(h_ratios, q, interpolation="higher")
    print("h_ratios:", ["{:.2f}:{:.2f}".format(a, b) for a, b in zip(q, res)])

    res = np.quantile(min_sizes, q, interpolation="higher")
    print("min_sizes:", ["{:.2f}:{:.2f}".format(a, b) for a, b in zip(q, res)])
