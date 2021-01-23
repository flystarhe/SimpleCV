import hiplot as hip
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
    base_sizes = [4, 8, 16, 32, 64]
    anchors = [s * x for s in scales for x in base_sizes]

    coco = load_json(coco_file)
    print("\n" + json.dumps(coco["categories"]) + "\n")
    imgs = {img["id"]: img["file_name"] for img in coco["images"]}

    data = []
    for ann in coco["annotations"]:
        w, h = [min(crop_size, x) for x in ann["bbox"][2:]]
        data.append({"file_name": imgs[ann["image_id"]], "id": ann["category_id"], "iou": best_iou(w, h, anchors, ratios),
                     "h_ratio": h / w, "h_ratio_log2": np.log2(h / w),
                     "area": w * h, "min_size": min(w, h)})

    hip.Experiment.from_iterable(data).display()
    return True
