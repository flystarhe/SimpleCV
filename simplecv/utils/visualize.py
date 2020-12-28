import cv2 as cv
import numpy as np
import os

from enum import Enum
from pathlib import Path


class Color(Enum):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)


def draw_bbox(anns, img, top, color_val):
    if not isinstance(anns, list):
        return None
    for i, ann in enumerate(anns, 1):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        left_bottom = (x, y - 2) if top else (x, y + h + 30)
        text = "{}: {}: {:.2f}: {}/{}={:.2f}".format(
            i, ann["label"], ann.get("score", 1.0), h, w, np.sqrt(h / w))
        cv.rectangle(img, (x, y), (x + w, y + h), color_val, thickness=1)
        cv.putText(img, text, left_bottom, cv.FONT_HERSHEY_COMPLEX, 1.0, color_val)
    return img


def save_image(out_dirs, file_name, img):
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, file_name), img)


def image_show(out_dirs, ori_file, dt, gt, dt_mask=None, gt_mask=None):
    if dt_mask is not None:
        dt = [d for d, flag in zip(dt, dt_mask) if flag == 1]

    if gt_mask is not None:
        gt = [g for g, flag in zip(gt, gt_mask) if flag == 1]

    dt = sorted(dt, key=lambda x: x["bbox"][1] // 100)

    out_img = cv.imread(ori_file, 1)
    text = Path(ori_file).parent.name
    cv.putText(out_img, text, (50, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, Color.blue.value)
    text = ",".join(sorted(set([d["label"] for d in dt])))
    cv.putText(out_img, text, (50, 60), cv.FONT_HERSHEY_COMPLEX, 1.0, Color.red.value)
    text = ",".join(sorted(set([g["label"] for g in gt])))
    cv.putText(out_img, text, (50, 90), cv.FONT_HERSHEY_COMPLEX, 1.0, Color.green.value)

    for i, d in enumerate(dt, 1):
        x, y, w, h = [int(v) for v in d["bbox"]]
        text = "{}: {}: {}: {:.2f}: {}/{}={:.2f}".format(
            i, (x, y), d["label"], d.get("score", 1.0), h, w, np.sqrt(h / w))
        cv.putText(out_img, text, (50, 90 + 30 * i), cv.FONT_HERSHEY_COMPLEX, 1.0, Color.blue.value)

    draw_bbox(dt, out_img, True, Color.red.value)
    draw_bbox(gt, out_img, False, Color.green.value)
    save_image(out_dirs, Path(ori_file).name, out_img)
