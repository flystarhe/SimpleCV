import os

import argparse
import cv2 as cv
import numpy as np
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from simplecv.beta.hej import io


################################################################
# Based on `mmdetection/mmdet/apis/inference.py`, insert code:
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import PIPELINES
from simplecv.mmdet_v2.ext_datasets import CocoDataset
from simplecv.mmdet_v2.ext_pipelines import RandomCrop, Resize2

DATASETS.register_module(name='CocoDataset', force=True, module=CocoDataset)
PIPELINES.register_module(name='RandomCrop', force=True, module=RandomCrop)
PIPELINES.register_module(name='Resize2', force=True, module=Resize2)
################################################################


def xyxy2xywh(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def split(size, patch_size):
    s = list(range(0, size - patch_size, patch_size))
    s.append(size - patch_size)
    return s


def norm_detector(model, img):
    result = inference_detector(model, img)
    if isinstance(result, tuple):
        bbox_result, _ = result
    else:
        bbox_result = result

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    return bboxes, labels


def patch_detector(patch_size, model, img):
    img_h, img_w, _ = img.shape

    if patch_size > min(img_h, img_w):
        return norm_detector(model, img)

    ys = split(img_h, patch_size)
    xs = split(img_w, patch_size)

    bboxes_list = []
    labels_list = []
    for y in ys:
        for x in xs:
            sub_img = img[y: y + patch_size, x: x + patch_size]
            bboxes, labels = norm_detector(model, sub_img)
            bboxes += np.array([x, y, x, y, 0])
            bboxes_list.append(bboxes)
            labels_list.append(labels)
    bboxes = np.vstack(bboxes_list)
    labels = np.concatenate(labels_list)
    return bboxes, labels


def test_imgs(img_list, config, checkpoint, patch_size):
    config = Config.fromfile(config)
    config.merge_from_dict(eval(os.environ.get("CFG_OPTIONS", "{}")))
    model = init_detector(config, checkpoint)
    classes = model.CLASSES

    results = []
    for file_name in img_list:
        img = cv.imread(file_name, 1)
        bboxes, labels = patch_detector(patch_size, model, img)

        dt = []
        for i in range(bboxes.shape[0]):
            xyxys, label = bboxes[i].tolist(), classes[labels[i]]
            dt.append(dict(bbox=xyxy2xywh(xyxys), xyxy=xyxys[:4], score=xyxys[4], label=label))
        results.append([file_name, dt])
    return results


def main(args):
    in_file = args.data
    img_list = io.load_json(in_file)
    outputs = test_imgs(img_list, args.config, args.checkpoint, args.patch_size)
    return io.save_pkl(outputs, in_file + ".out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("data", type=str, help="json file path")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("checkpoint", type=str, help="checkpoint file path")
    parser.add_argument("patch_size", type=int, default=999999, help="json file path")
    args = parser.parse_args()
    print(main(args))
