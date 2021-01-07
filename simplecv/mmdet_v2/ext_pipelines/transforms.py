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

    def __init__(self, height, width, p=1.0):
        safe_area = min(width // 3, 96) ** 2
        self.nonignore_area = safe_area
        self.height = height
        self.width = width
        self.p = p

    def _index_selection(self, labels):
        counter = defaultdict(list)
        for ind, label in enumerate(labels):
            counter[label].append(ind)

        key = np.random.choice(list(counter.keys()))
        return np.random.choice(counter[key])

    def _get_patch(self, x_min, y_min, x_max, y_max, x_pad, y_pad):
        x0 = np.random.randint(x_min - x_pad, x_max + x_pad - self.width)
        y0 = np.random.randint(y_min - y_pad, y_max + y_pad - self.height)
        return np.array([x0, y0, x0 + self.width, y0 + self.height], dtype="int64")

    def _clip_bboxes(self, patch, bboxes):
        """
        Args:
            patch (ndarray): shape (4,)
            bboxes (ndarray): shape (k, 4)
        """
        bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
        bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
        bboxes -= np.tile(patch[:2], 2)
        return bboxes

    def _check_bboxes(self, src_bboxes, dst_bboxes):
        src_w = src_bboxes[:, 2] - src_bboxes[:, 0]
        src_h = src_bboxes[:, 3] - src_bboxes[:, 1]
        src_area = src_w * src_h

        dst_w = dst_bboxes[:, 2] - dst_bboxes[:, 0]
        dst_h = dst_bboxes[:, 3] - dst_bboxes[:, 1]
        dst_area = dst_w * dst_h

        s1 = (dst_area >= src_area * 0.7)
        s2 = (dst_area >= self.nonignore_area)
        s3 = (dst_h >= src_h * 0.7) * (dst_w >= src_h * 1.5)
        s4 = (dst_w >= src_w * 0.7) * (dst_h >= src_w * 1.5)

        dst = s1 + s2 + s3 + s4
        drop = np.logical_not(dst)
        inner = (dst_w > 1) * (dst_h > 1)
        return (dst * inner), (drop * inner)

    def _crop_and_paste(self, patch, img):
        x1, y1, x2, y2 = patch
        img_h, img_w, _ = img.shape

        p1, x1 = (-x1, 0) if x1 < 0 else (0, x1)
        q1, y1 = (-y1, 0) if y1 < 0 else (0, y1)

        p2, x2 = (img_w - x2, img_w) if x2 > img_w else (self.width, x2)
        q2, y2 = (img_h - y2, img_h) if y2 > img_h else (self.height, y2)

        dst_img = np.zeros((self.height, self.width, 3), dtype=img.dtype)
        dst_img[q1: q2, p1: p2] = img[y1: y2, x1: x2]
        return dst_img

    def __call__(self, results):
        img, bboxes, labels = [results[k] for k in ("img", "gt_bboxes", "gt_labels")]
        img_h, img_w, img_c = img.shape
        assert img_c == 3

        if bboxes.shape[0] == 0:
            x_min, y_min, x_max, y_max = 0, 0, img_w, img_h
            x_pad, y_pad = self.width // 2 - 64, self.height // 2 - 64
            patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

            dst_img = self._crop_and_paste(patch, img)

            cx, cy = self.width // 2, self.height // 2
            x1, y1, x2, y2 = cx - 32, cy - 32, cx + 32, cy + 32
            dst_img[y1: y2, x1: x2] = dst_img[y1: y2, x1: x2] - 50
            dst_bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)  # man-made object
            dst_labels = np.array([0], dtype=np.int64)  # set the man-made object category in 1st

            results["img"] = dst_img
            results["img_shape"] = dst_img.shape
            results["ori_shape"] = dst_img.shape
            results["pad_shape"] = dst_img.shape
            results["gt_bboxes"] = dst_bboxes
            results["gt_labels"] = dst_labels
            return results

        bboxes = bboxes.astype("int64")
        index = self._index_selection(labels)
        x_min, y_min, x_max, y_max = bboxes[index]
        x_pad, y_pad = self.width // 3 * 2, self.height // 3 * 2
        patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

        dst_img = self._crop_and_paste(patch, img)

        dst_bboxes = self._clip_bboxes(patch, bboxes.copy())
        dst_mask, drop_mask = self._check_bboxes(bboxes, dst_bboxes)
        for x1, y1, x2, y2 in dst_bboxes[drop_mask]:
            dst_img[y1: y2, x1: x2] = 0
        dst_bboxes = dst_bboxes.astype("float32")
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
