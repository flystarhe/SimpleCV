import os

import argparse
import cv2 as cv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel import scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from simplecv.beta.hej import io


################################################################
# Based on `mmdetection/mmdet/apis/inference.py`, insert code:
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import PIPELINES
from simplecv.mmdet_v2.ext_datasets import CocoDataset
from simplecv.mmdet_v2.ext_pipelines import RandomCrop, Resize2

DATASETS.register_module(name="CocoDataset", force=True, module=CocoDataset)
PIPELINES.register_module(name="RandomCrop", force=True, module=RandomCrop)
PIPELINES.register_module(name="Resize2", force=True, module=Resize2)
################################################################


def xyxy2xywh(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def split(size, patch_size, overlap=64):
    if patch_size >= size:
        return [0]

    s = list(range(0, size - patch_size, patch_size - overlap))
    s.append(size - patch_size)
    return s


def inference_detector(model, imgs, device=None, test_pipeline=None):
    # imgs (list[ndarray]): `[cv.imread(file_name, 1), ...]`
    if device is None:
        device = next(model.parameters()).device

    if test_pipeline is None:
        cfg = model.cfg.copy()
        test_pipeline = cfg.data.test.pipeline
        test_pipeline[0].type = "LoadImageFromWebcam"
        test_pipeline = Compose(test_pipeline)

    data = [test_pipeline(dict(img=img)) for img in imgs]
    data = collate(data, samples_per_gpu=len(data))

    # just get the actual data from DataContainer
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    data = scatter(data, [device])[0]
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return results


def _bbox_result(result):
    if isinstance(result, tuple):
        result = result[0]

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    return bboxes, labels


def patch_detector(patch_size, model, img, device=None, test_pipeline=None):
    img_h, img_w, _ = img.shape
    ys = split(img_h, patch_size)
    xs = split(img_w, patch_size)

    imgs = [img[y: y + patch_size, x: x + patch_size] for y in ys for x in xs]
    results = inference_detector(model, imgs, device, test_pipeline)
    results = [_bbox_result(r) for r in results]
    bboxes_list, labels_list = zip(*results)

    bboxes = np.vstack(bboxes_list)
    labels = np.concatenate(labels_list)
    return bboxes, labels


def test_imgs(img_list, config, checkpoint, patch_size):
    config = Config.fromfile(config)
    config.merge_from_dict(eval(os.environ.get("CFG_OPTIONS", "{}")))
    model = init_detector(config, checkpoint, device="cuda:0")
    device = next(model.parameters()).device
    classes = model.CLASSES

    cfg = model.cfg.copy()
    test_pipeline = cfg.data.test.pipeline
    test_pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(test_pipeline)

    results = []
    for file_name in img_list:
        img = cv.imread(file_name, 1)
        bboxes, labels = patch_detector(patch_size, model, img, device, test_pipeline)

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
    parser.add_argument("patch_size", type=int, default=999999, help="patch size")
    args = parser.parse_args()
    print(main(args))
