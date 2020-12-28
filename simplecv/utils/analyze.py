import os

import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

from simplecv.beta.hej import io
from simplecv.utils import nms
from simplecv.utils import increment_path
from simplecv.utils import visualize as viz


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    elif "*" in data:
        return data["*"]
    else:
        return val


def list2str(lst, fmt="{}"):
    return ",".join([fmt.format(i) for i in lst])


def format_confusion_matrix(y_true, y_pred, labels, title):
    labels = sorted(set(labels + y_true + y_pred))
    table_data = confusion_matrix(y_true, y_pred, labels=labels)
    n_good, n_total = table_data.diagonal().sum(), table_data.sum()

    lines = []
    pr_p, pr_r = [], []
    lines.append("\n{}\n{},".format(title, "Y:Y^") + list2str(labels))
    for i in range(table_data.shape[0]):
        lines.append("{},".format(labels[i]) + list2str(table_data[i]))
        rsum, csum = np.sum(table_data[i, :]), np.sum(table_data[:, i])
        pr_p.append("{:.2f}%".format(table_data[i, i] / csum * 100 if csum > 0 else 0))
        pr_r.append("{:.2f}%".format(table_data[i, i] / rsum * 100 if rsum > 0 else 0))
    lines.append("{},".format("P(%)") + list2str(pr_p) + "\n{},".format("R(%)") + list2str(pr_r))
    lines.append("good,{},total,{},good/total,{:.2f}%".format(n_good, n_total, n_good / n_total * 100))
    return lines


def matrix_analysis_object(results, score_thr, out_file=None, **kwargs):
    """Matrix analysis by object.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.utils.translate.trans_test_results()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        out_file (str): Save to file path, default `None`.
    Returns:
        lines (list): List of CSV lines.
    """
    nms_thr = kwargs.get("nms_thr", 0.3)
    clean_mode = kwargs.get("clean_mode", "min")
    match_mode = kwargs.get("match_mode", "iou")
    pos_iou_thr = kwargs.get("pos_iou_thr", 0.3)
    min_pos_iou = kwargs.get("min_pos_iou", pos_iou_thr - 0.1)

    y_true, y_pred = [], []
    n_dt, n_dt_match, n_gt, n_gt_match = 0, 0, 0, 0
    bads_tail = ["file_name,dt,gt,hard,missed,false_pos"]
    for file_name, _, _, dt, gt in results:
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        dt = nms.clean_by_bbox(dt, nms_thr=nms_thr, mode=clean_mode)
        ious = nms.bbox_overlaps(dt, gt, mode=match_mode)

        n_hard = 0
        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                if ious[i, j] >= pos_iou_thr:
                    d_label = dt[i]["label"]
                    g_label = gt[j]["label"]
                    y_pred.append(d_label)
                    y_true.append(g_label)
                    exclude_i.add(i)
                    if d_label != g_label:
                        n_hard += 1

            for j, i in enumerate(ious.argmax(axis=0)):
                if ious[i, j] >= min_pos_iou:
                    exclude_j.add(j)
        # for i, j in zip(*np.where(ious >= min_pos_iou))-- fast
        # for i, j in np.argwhere(ious >= min_pos_iou) -- is slow

        n_false_pos = 0
        for i in range(len(dt)):
            if i not in exclude_i:
                d_label = dt[i]["label"]
                y_pred.append(d_label)
                y_true.append("none")
                n_false_pos += 1

        n_missed = 0
        for j in range(len(gt)):
            if j not in exclude_j:
                g_label = gt[j]["label"]
                y_pred.append("none")
                y_true.append(g_label)
                n_missed += 1

        n_dt, n_dt_match = n_dt + len(dt), n_dt_match + len(exclude_i)
        n_gt, n_gt_match = n_gt + len(gt), n_gt_match + len(exclude_j)
        bads_tail.append("{},{},{},{},{},{}".format(file_name, len(dt), len(gt), n_hard, n_missed, n_false_pos))

    n_missed = sum([1 for a, b in zip(y_pred, y_true) if a == "none"])
    n_false_pos = sum([1 for a, b in zip(y_pred, y_true) if b == "none"])
    title = "{}\nCM[object]\ngt,{}\ndt,{}\ngt_match,{}\ndt_match,{}\nmissed,{}\nfalse_pos,{}".format(
        out_file, n_gt, n_dt, n_gt_match, n_dt_match, n_missed, n_false_pos)
    lines = format_confusion_matrix(y_true, y_pred, [], title)
    if out_file is not None:
        io.save_csv(lines + bads_tail, out_file)
    print("\n".join(lines))
    return bads_tail[1:]


def matrix_analysis_image(results, score_thr, out_file=None):
    """Matrix analysis by image.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.utils.translate.trans_test_results()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        out_file (str): Save to file path, default `None`.
    Returns:
        lines (list): List of CSV lines.
    """
    y_true, y_pred = [], []
    for _, target, predict, _, _ in results:
        if predict["score"] >= get_val(score_thr, predict["label"], 0.3):
            y_pred.append(predict["label"])
            y_true.append(target)
    n_total, n_cover = len(results), len(y_true)
    title = "{}\nCM[image]\ntotal,{}\ncover,{}\nsay {:.2f}%".format(out_file, n_total, n_cover, n_cover / n_total * 100)
    lines = format_confusion_matrix(y_true, y_pred, [], title)
    if out_file is not None:
        io.save_csv(lines, out_file)
    print("\n".join(lines))
    return lines


def display_dataset(results, score_thr, output_dir, simple=False, **kwargs):
    """Show model prediction results, none gt.

    Args:
        results (list): List of `(file_name, none, none, dt, none)`.
            Via `simplecv.mmdet_v2.py_test.test_dir()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        output_dir (str): Directory where the draft are saved.
        simple (bool): Whether to simplify the draft image.
        show (bool): Whether to save the bbox-picture.
    Returns:
        None.
    """
    nms_thr = kwargs.get("nms_thr", 0.3)
    clean_mode = kwargs.get("clean_mode", "min")
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        results = io.load_pkl(results)

    for file_name, _, _, dt, gt in results:
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if simple:
            dt = nms.clean_by_bbox(dt, nms_thr=nms_thr, mode=clean_mode)

        viz.image_show([os.path.join(output_dir, "images-pred")], file_name, dt, gt, None, None)
    return str(output_dir)


def display_hardmini(results, score_thr, output_dir, simple=True, **kwargs):
    """Hardmini focus `FP + FN`.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.utils.translate.trans_test_results()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        output_dir (str): Directory where the draft are saved.
        simple (bool): Whether to simplify the draft image.
        show (bool): Whether to save the bbox-picture.
    Returns:
        None.
    """
    show = kwargs.get("show", False)
    nms_thr = kwargs.get("nms_thr", 0.3)
    clean_mode = kwargs.get("clean_mode", "min")
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        results = io.load_pkl(results)

    out_file = os.path.join(output_dir, "readme.csv")
    bads_tail = matrix_analysis_object(results, score_thr, out_file, **kwargs)

    if not show:
        return str(output_dir)

    for (file_name, _, _, dt, gt), line in zip(results, bads_tail):
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if simple:
            dt = nms.clean_by_bbox(dt, nms_thr=nms_thr, mode=clean_mode)

        tt = line.split(",")[-3:]
        ss = ["hard", "missed", "false_pos"]
        ss = [s for t, s in zip(tt, ss) if t != "0"]

        if ss:
            out_dirs = [os.path.join(output_dir, s) for s in ss]
            viz.image_show(out_dirs, file_name, dt, gt, None, None)
    return str(output_dir)
