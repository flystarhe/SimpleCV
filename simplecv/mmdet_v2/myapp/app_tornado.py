import cv2 as cv
import numpy as np
import os
import sys
import time
import tornado.ioloop
import tornado.web
import traceback

from mmdet.apis import init_detector, inference_detector
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


def split(size, patch_size, overlap=64):
    s = list(range(0, size - patch_size, patch_size - overlap))
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
classes = None
patch_size = 999999
clean_mode = "min"
clean_param = 0.3
cached = Cache()


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        try:
            global model, classes, patch_size, clean_mode, clean_param

            img = cv.imread(self.get_argument("image"), 1)
            bboxes, labels = patch_detector(patch_size, model, img)

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
    classes = model.CLASSES

    if len(sys.argv) >= 5:
        patch_size = sys.argv[4]

    if len(sys.argv) >= 7:
        clean_mode, clean_param = sys.argv[5], sys.argv[6]

    cached.add(argv=sys.argv)
    patch_size = int(patch_size)
    clean_param = float(clean_param)

    make_app().listen(int(port))
    tornado.ioloop.IOLoop.current().start()
