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


def bboxes2dt(bbox_, code_):
    dt = []
    for i in range(bbox_.shape[0]):
        xyxys = bbox_[i].tolist()
        ann = dict(xyxy=xyxys[:4], score=xyxys[4], label=code_)
        dt.append(ann)
    return dt


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
cached = Cache()
clean_thr = 0.1
clean_mode = "min"


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        try:
            global model, classes
            global clean_thr, clean_mode
            image_path = self.get_argument("image")
            result = inference_detector(model, image_path)
            if isinstance(result, tuple):
                bboxes, _ = result
            else:
                bboxes = result

            dt = []
            for bbox_, code_ in zip(bboxes, classes):
                dt.extend(bboxes2dt(bbox_, code_))

            dt = clean_by_bbox(dt, clean_thr, clean_mode)
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
    config_file, checkpoint_file = sys.argv[2], sys.argv[3]
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    classes = model.CLASSES

    if len(sys.argv) == 6:
        clean_thr, clean_mode = float(sys.argv[4]), sys.argv[5]

    cached.add(argv=sys.argv)

    app = make_app()
    app.listen(sys.argv[1])
    tornado.ioloop.IOLoop.current().start()
