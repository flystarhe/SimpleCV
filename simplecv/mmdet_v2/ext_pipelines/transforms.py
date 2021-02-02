"""Transforms for mmdet2.x
"""
import cv2 as cv
import numpy as np
from collections import defaultdict


class RandomCrop(object):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability.
    Targets:
        image, bboxes
    Image types:
        uint8,
    """

    def __init__(self, height, width, **kw):
        area = (height * width) * 0.5
        self.nonignore = area
        self.height = height
        self.width = width

    def _index_selection(self, labels):
        counter = defaultdict(list)
        for index, label in enumerate(labels):
            counter[label].append(index)

        key = np.random.choice(list(counter.keys()))
        return np.random.choice(counter[key])

    def _get_patch(self, x_min, y_min, x_max, y_max, x_pad, y_pad):
        x0 = np.random.randint(x_min - x_pad, x_max + x_pad - self.width)
        y0 = np.random.randint(y_min - y_pad, y_max + y_pad - self.height)
        return x0, y0, x0 + self.width, y0 + self.height

    def _crop_and_paste(self, img, patch):
        img_h, img_w, img_c = img.shape
        x1, y1, x2, y2 = patch

        p1, x1 = (0, x1) if x1 >= 0 else (-x1, 0)
        q1, y1 = (0, y1) if y1 >= 0 else (-y1, 0)

        p2, x2 = (self.width, x2) if x2 <= img_w else (img_w - x2, img_w)
        q2, y2 = (self.height, y2) if y2 <= img_h else (img_h - y2, img_h)

        dst_img = np.zeros((self.height, self.width, img_c), dtype=img.dtype)
        dst_img[q1: q2, p1: p2] = img[y1: y2, x1: x2]
        return dst_img

    def _clip_bboxes(self, bboxes, patch):
        """
        Args:
            bboxes (ndarray): shape `(k, 4)`, dtype `np.float32`
            patch (ndarray): shape `(4,)` or tuple `(x1, y1, x2, y2)`
        """
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], patch[0], patch[2])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], patch[1], patch[3])
        bboxes -= np.tile(patch[:2], 2)
        return bboxes

    def _check_bboxes(self, src_bboxes, dst_bboxes):
        src_w = src_bboxes[:, 2] - src_bboxes[:, 0]
        src_h = src_bboxes[:, 3] - src_bboxes[:, 1]
        src_area = src_w * src_h

        dst_w = dst_bboxes[:, 2] - dst_bboxes[:, 0]
        dst_h = dst_bboxes[:, 3] - dst_bboxes[:, 1]
        dst_area = dst_w * dst_h

        inner = (dst_w >= 4) * (dst_h >= 4) * (dst_area >= 48)
        x = np.clip(np.sqrt(src_area), 32, None)
        x = 0.5 + 0.5 * 32 / x

        s1 = (dst_area >= src_area * x)
        s2 = (dst_area >= self.nonignore)
        s3 = (dst_w >= src_w) * (dst_h >= src_w * 2.0)
        s4 = (dst_h >= src_h) * (dst_w >= src_h * 2.0)
        ss = s1 + s2 + s3 + s4

        return ss * inner, np.logical_not(ss) * inner

    def __call__(self, results):
        img, bboxes, labels = [results[k] for k in ("img", "gt_bboxes", "gt_labels")]
        img_h, img_w, img_c = img.shape
        assert img_c == 3

        if bboxes.shape[0] == 0:
            x_min, y_min, x_max, y_max = 0, 0, img_w, img_h

            x_pad = max(self.width - img_w, 64)
            y_pad = max(self.height - img_h, 64)
            patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

            dst_img = self._crop_and_paste(img, patch)

            cx, cy = self.width // 2, self.height // 2
            x1, y1, x2, y2 = cx - 32, cy - 32, cx + 32, cy + 32
            dst_img[y1: y2, x1: x2] = dst_img[y1: y2, x1: x2] + 128
            dst_bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)  # man-made object
            dst_labels = np.array([0], dtype=np.int64)  # set the man-made object category in 1st

            results["img"] = dst_img
            results["img_shape"] = dst_img.shape
            results["ori_shape"] = dst_img.shape
            results["pad_shape"] = dst_img.shape
            results["gt_bboxes"] = dst_bboxes
            results["gt_labels"] = dst_labels
            return results

        index = self._index_selection(labels)
        x_min, y_min, x_max, y_max = map(int, bboxes[index])

        x_pad = max(self.width - (x_max - x_min), 64)
        y_pad = max(self.height - (y_max - y_min), 64)
        patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

        dst_img = self._crop_and_paste(img, patch)
        dst_bboxes = self._clip_bboxes(bboxes.copy(), patch)
        dst_mask, drop_mask = self._check_bboxes(bboxes, dst_bboxes)
        for x1, y1, x2, y2 in dst_bboxes[drop_mask]:
            dst_img[y1: y2, x1: x2] = 0
        dst_bboxes = dst_bboxes[dst_mask]
        dst_labels = labels[dst_mask]

        results["img"] = dst_img
        results["img_shape"] = dst_img.shape
        results["ori_shape"] = dst_img.shape
        results["pad_shape"] = dst_img.shape
        results["gt_bboxes"] = dst_bboxes
        results["gt_labels"] = dst_labels
        return results


class Resize2(object):

    def __init__(self, test_mode=False, ratio_range=(0.8, 1.2), **kw):
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
