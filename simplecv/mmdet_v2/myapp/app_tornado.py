import cv2 as cv
import numpy as np
import os
import sys
import time
import tornado.ioloop
import tornado.web
import traceback

import torch
from mmcv.parallel import collate
from mmcv.parallel import scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import PIPELINES
from app_nms import clean_by_bbox


os.environ["MKL_THREADING_LAYER"] = "GNU"


class Resize2(object):

    def __init__(self, test_mode=False, ratio_range=(0.8, 1.2), **kwargs):
        self.test_mode = test_mode
        self.ratio_range = ratio_range

    def __call__(self, results):
        img = results["img"]

        if self.test_mode:
            results["img_shape"] = img.shape
            results["pad_shape"] = img.shape
            results["scale_factor"] = 1.0
            results["keep_ratio"] = True
            return results

        h, w = img.shape[:2]
        a, b = self.ratio_range

        scale_factor = (b - a) * np.random.random_sample() + a
        new_size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
        img = cv.resize(img, new_size, dst=None, interpolation=cv.INTER_LINEAR)

        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = True

        # resize bboxes
        bboxes = results["gt_bboxes"]
        img_shape = results["img_shape"]
        bboxes = bboxes * results["scale_factor"]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        results["gt_bboxes"] = bboxes
        return results


PIPELINES.register_module(name="Resize2", force=True, module=Resize2)


def xyxy2xywh(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def split(size, patch_size, overlap=128):
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


def _bbox_result(result, offset=None):
    if isinstance(result, tuple):
        result = result[0]

    if offset is None:
        offset = 0.

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    return bboxes + offset, labels


def patch_detector(patch_size, model, img, device=None, test_pipeline=None):
    img_h, img_w = img.shape[:2]
    ys = split(img_h, patch_size)
    xs = split(img_w, patch_size)

    results = []
    offsets = []
    batch_size = 8
    xys = [(y, x) for y in ys for x in xs]
    for i in range(0, len(xys), batch_size):
        imgs = [img[y: y + patch_size, x: x + patch_size] for y, x in xys[i: i + batch_size]]
        offsets.extend([np.array([x, y, x, y, 0.]) for y, x in xys[i: i + batch_size]])
        results.extend(inference_detector(model, imgs, device, test_pipeline))

    results = [_bbox_result(result, offset) for result, offset in zip(results, offsets)]
    bboxes_list, labels_list = zip(*results)

    bboxes = np.vstack(bboxes_list)
    labels = np.concatenate(labels_list)
    return bboxes, labels


class Cache(object):

    def __init__(self, limit=30):
        self.limit = limit
        self.data = ["service is running..."]

    def add(self, **kwargs):
        txt = ", ".join(["{}: {}".format(k, v) for k, v in kwargs.items()])
        if len(self.data) >= self.limit:
            self.data = self.data[:(self.limit // 2)]
        self.data.append(txt)

    def logs(self):
        return "<br />".join(self.data)


model = None
device = None
classes = None
test_pipeline = None
patch_size = 999999
clean_mode = "min"
clean_param = 0.1
cached = Cache()


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        try:
            global model, device, classes, test_pipeline, patch_size, clean_mode, clean_param

            img = cv.imread(self.get_argument("image"), 1)
            bboxes, labels = patch_detector(patch_size, model, img, device, test_pipeline)

            dt = []
            for i in range(bboxes.shape[0]):
                xyxys, label = bboxes[i].tolist(), classes[labels[i]]
                dt.append(dict(bbox=xyxy2xywh(xyxys), xyxy=xyxys[:4], score=xyxys[4], label=label))

            dt = clean_by_bbox(dt, clean_mode, clean_param)
            data = [d["xyxy"] + [d["label"], d["score"]] for d in dt]
            res = {"status": 0, "time": int(time.time()), "data": data}
        except Exception:
            err = traceback.format_exc()
            res = {"status": 1, "time": int(time.time()), "data": err}
        self.finish(res)
        cached.add(main=res)


class LogHandler(tornado.web.RequestHandler):

    def get(self):
        self.finish(cached.logs())


def make_app():
    return tornado.web.Application([
        (r"/main", MainHandler),
        (r"/logs", LogHandler),
    ])


if __name__ == "__main__":
    port = sys.argv[1]
    config_file = sys.argv[2]
    checkpoint_file = sys.argv[3]

    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    device = next(model.parameters()).device
    classes = model.CLASSES

    cfg = model.cfg.copy()
    test_pipeline = cfg.data.test.pipeline
    test_pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(test_pipeline)

    if len(sys.argv) >= 5:
        patch_size = sys.argv[4]

    if len(sys.argv) >= 7:
        clean_mode, clean_param = sys.argv[5], sys.argv[6]

    cached.add(argv=sys.argv)
    patch_size = int(patch_size)
    clean_param = float(clean_param)

    make_app().listen(int(port))
    tornado.ioloop.IOLoop.current().start()
