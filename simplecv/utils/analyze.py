import os
import shutil

import hiplot as hip
import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from sklearn.metrics import confusion_matrix

from simplecv.beta.hej import io
from simplecv.utils import nms
from simplecv.utils import increment_path
from simplecv.utils import visualize as viz


def json_type(v):
    try:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, Iterable):
            return [json_type(i) for i in v]
        return float(v)
    except Exception:
        print("Unknown type:", type(v), v)
    return v


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    if "*" in data:
        return data["*"]
    return val


def gen_multi_labels(anns, simple=True):
    cache = defaultdict(list)
    for ann in anns:
        cache[ann["label"]].append(ann)

    best_result = [max(v, key=lambda x: x["score"]) for v in cache.values()]
    if simple:
        best_result = [best["label"] for best in best_result]
    return best_result


def agent_split(x, q):
    if len(x) == 0:
        return 0, [0 for _ in q]
    return len(x), np.quantile(x, q, interpolation="higher")


def list2str(lst, fmt="{}"):
    return ",".join([fmt.format(i) for i in lst])


def format_confusion_matrix(y_true, y_pred, labels, title):
    labels = sorted(set(labels + y_true + y_pred))
    table_data = confusion_matrix(y_true, y_pred, labels=labels)

    lines = []
    pr_p, pr_r = [], []
    lines.append("\n{}\n{},".format(title, "Y:Y^") + list2str(labels))
    for i in range(table_data.shape[0]):
        lines.append("{},".format(labels[i]) + list2str(table_data[i]))
        rsum, csum = np.sum(table_data[i, :]), np.sum(table_data[:, i])
        pr_p.append("{:.2f}%".format(table_data[i, i] / csum * 100 if csum > 0 else 0))
        pr_r.append("{:.2f}%".format(table_data[i, i] / rsum * 100 if rsum > 0 else 0))
    lines.append("{},".format("P(%)") + list2str(pr_p) + "\n{},".format("R(%)") + list2str(pr_r))

    n_good, n_total = table_data.diagonal().sum(), table_data.sum()
    lines.append("good,{},total,{},good/total,{:.2f}%".format(n_good, n_total, n_good / n_total * 100))

    if len(labels) > 1:
        table_data = table_data[:-1, :-1]
        n_good, n_total = table_data.diagonal().sum(), table_data.sum()
        lines.append("good,{},total,{},good/total,{:.2f}%".format(n_good, n_total, n_good / n_total * 100))
    return lines


def format_missed_false_pos(total_gt, total_pos, score_missed, score_false_pos, q=20):
    labels = sorted(set(list(total_gt.keys()) + list(total_pos.keys())))

    lines = []
    if isinstance(q, int):
        q = [i / q for i in range(0, q + 1)]

    for label in labels:
        i_missed, q_missed = agent_split(score_missed[label], q)
        i_false_pos, q_false_pos = agent_split(score_false_pos[label], q)
        lines.append("\nGROUP,{},{},{},".format("LABEL", "TOTAL", "FOCUS") + list2str(q, "{:.2f}"))
        lines.append("GT,{},{},{},".format(label, total_gt[label], i_missed) + list2str(q_missed, "{:.2f}"))
        lines.append("POS,{},{},{},".format(label, total_pos[label], i_false_pos) + list2str(q_false_pos, "{:.2f}"))
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
    clean_mode = kwargs.get("clean_mode", "min")
    clean_param = kwargs.get("clean_param", 0.1)
    match_mode = kwargs.get("match_mode", "iou")
    pos_iou_thr = kwargs.get("pos_iou_thr", 0.3)
    min_pos_iou = kwargs.get("min_pos_iou", 0.1)

    total_gt = defaultdict(int)
    total_pos = defaultdict(int)
    score_missed = defaultdict(list)
    score_false_pos = defaultdict(list)

    y_true, y_pred = [], []
    bads_tail = ["\nfile_name,dt,gt,hard,missed,false_pos"]
    n_dt, n_gt, n_hard, n_missed, n_false_pos = 0, 0, 0, 0, 0
    for file_name, _, _, dt, gt in results:
        dt = nms.clean_by_bbox(dt, clean_mode, clean_param)
        ious = nms.bbox_overlaps(dt, gt, match_mode)

        i_hard = 0
        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                d_label = dt[i]["label"]
                d_score = dt[i]["score"]
                g_label = gt[j]["label"]
                total_pos[d_label] += 1

                if ious[i, j] >= pos_iou_thr:
                    if d_score >= get_val(score_thr, d_label, 0.3):
                        exclude_i.add(i)
                        y_pred.append(d_label)
                        y_true.append(g_label)
                        if d_label != g_label:
                            i_hard += 1
                    if d_label != g_label:
                        score_false_pos[d_label].append(d_score)
                else:
                    score_false_pos[d_label].append(d_score)

            safe_index = set()
            for i, j in zip(*np.where(ious >= min_pos_iou)):
                if dt[i]["score"] >= get_val(score_thr, dt[i]["label"], 0.3):
                    safe_index.add(j)

            for j, i in enumerate(ious.argmax(axis=0)):
                d_label = dt[i]["label"]
                d_score = dt[i]["score"]
                g_label = gt[j]["label"]
                total_gt[g_label] += 1

                if ious[i, j] >= min_pos_iou:
                    if d_score >= get_val(score_thr, d_label, 0.3):
                        exclude_j.add(j)
                    elif j in safe_index:
                        exclude_j.add(j)
                    else:
                        score_missed[d_label].append(d_score)
                else:
                    score_missed[g_label].append(0.0)

        i_false_pos = 0
        for i, d in enumerate(dt):
            if i not in exclude_i:
                y_pred.append(d["label"])
                y_true.append("none")
                i_false_pos += 1

        i_missed = 0
        for j, g in enumerate(gt):
            if j not in exclude_j:
                y_pred.append("none")
                y_true.append(g["label"])
                i_missed += 1

        i_dt, i_gt = len(dt), len(gt)
        bads_tail.append("{},{},{},{},{},{}".format(
            file_name, i_dt, i_gt, i_hard, i_missed, i_false_pos))

        n_dt, n_gt = n_dt + i_dt, n_gt + i_gt
        n_hard, n_missed, n_false_pos = n_hard + i_hard, n_missed + i_missed, n_false_pos + i_false_pos

    title = "{}\nCM[object]\ndt,{}\ngt,{}\nhard,{}\nmissed,{}\nfalse_pos,{}".format(
        out_file, n_dt, n_gt, n_hard, n_missed, n_false_pos)
    lines = format_confusion_matrix(y_true, y_pred, [], title)
    lines += format_missed_false_pos(total_gt, total_pos, score_missed, score_false_pos)
    if out_file is not None:
        io.save_csv(lines + bads_tail, out_file)
    print("\n".join(lines))
    return bads_tail[1:]


def matrix_analysis_image(results, score_thr, out_file=None, **kwargs):
    """Matrix analysis by image.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.utils.translate.trans_test_results()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        out_file (str): Save to file path, default `None`.
    Returns:
        lines (list): List of CSV lines.
    """
    single_cls = kwargs.get("single_cls", True)

    assert not single_cls or results[0][1] is not None

    y_true, y_pred = [], []
    bads_tail = ["\nfile_name,dt,gt,hard,missed,false_pos"]
    n_dt, n_gt, n_hard, n_missed, n_false_pos = 0, 0, 0, 0, 0
    for file_name, target, predict, dt, gt in results:
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]

        if single_cls:
            flag = predict["score"] >= get_val(score_thr, predict["label"], 0.3)
            dt_label = predict["label"] if flag else "none"
            gt_label = target if gt else "none"
            excluded = set(["none"])

            dt_labels = set([dt_label]) - excluded
            gt_labels = set([gt_label]) - excluded
        else:
            dt_labels = set(gen_multi_labels(dt))
            gt_labels = set(gen_multi_labels(gt))

        for label in (dt_labels & gt_labels):
            y_pred.append(label)
            y_true.append(label)

        i_false_pos = 0
        for label in (dt_labels - gt_labels):
            y_pred.append(label)
            y_true.append("none")
            i_false_pos += 1

        i_missed = 0
        for label in (gt_labels - dt_labels):
            y_pred.append("none")
            y_true.append(label)
            i_missed += 1

        i_dt, i_gt = len(dt_labels), len(gt_labels)
        i_hard = 1 if i_missed > 0 and i_false_pos > 0 else 0
        bads_tail.append("{},{},{},{},{},{}".format(
            file_name, i_dt, i_gt, i_hard, i_missed, i_false_pos))

        n_dt, n_gt = n_dt + i_dt, n_gt + i_gt
        n_hard, n_missed, n_false_pos = n_hard + i_hard, n_missed + i_missed, n_false_pos + i_false_pos

    title = "{}\nCM[image]\ndt,{}\ngt,{}\nhard,{}\nmissed,{}\nfalse_pos,{}".format(
        out_file, n_dt, n_gt, n_hard, n_missed, n_false_pos)
    lines = format_confusion_matrix(y_true, y_pred, [], title)
    if out_file is not None:
        io.save_csv(lines + bads_tail, out_file)
    print("\n".join(lines))
    return bads_tail[1:]


def display_hardmini(results, score_thr, output_dir, **kwargs):
    """Hardmini focus `FP + FN`.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.utils.translate.trans_test_results()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        output_dir (str): Directory where the draft are saved.
    Returns:
        None.
    """
    show = kwargs.get("show", False)
    simple = kwargs.get("simple", True)
    clean_mode = kwargs.get("clean_mode", "min")
    clean_param = kwargs.get("clean_param", 0.1)
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        shutil.copy(results, output_dir)
        results = io.load_pkl(results)

    out_file = os.path.join(output_dir, "readme.csv")
    bads_tail = matrix_analysis_object(results, score_thr, out_file, **kwargs)

    if not show:
        return str(out_file)

    for (file_name, _, _, dt, gt), line in zip(results, bads_tail):
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if simple:
            dt = nms.clean_by_bbox(dt, clean_mode, clean_param)

        tt = line.split(",")[-3:]
        ss = ["hard", "missed", "false_pos"]
        ss = [s for t, s in zip(tt, ss) if t != "0"]

        if ss:
            out_dirs = [os.path.join(output_dir, s) for s in ss]
            viz.image_show(out_dirs, file_name, dt, gt, None, None)

            ann_path = Path(file_name).with_suffix(".json")
            for out_dir in out_dirs:
                shutil.copy(ann_path, out_dir)
    return str(output_dir)


def hiplot_analysis_object(results, score_thr, **kwargs):
    clean_mode = kwargs.get("clean_mode", "min")
    clean_param = kwargs.get("clean_param", 0.1)
    match_mode = kwargs.get("match_mode", "iou")
    min_pos_iou = kwargs.get("min_pos_iou", 1e-5)

    if isinstance(results, str):
        results = io.load_pkl(results)

    vals = []
    for file_name, _, _, dt, gt in results:
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        dt = nms.clean_by_bbox(dt, clean_mode, clean_param)
        ious = nms.bbox_overlaps(dt, gt, match_mode)

        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                iou = ious[i, j]
                d, g = dt[i], gt[j]
                if iou >= min_pos_iou:
                    a = [d["label"], d["score"]] + d["bbox"][2:]
                    b = [g["label"], g["score"]] + g["bbox"][2:]
                    vals.append([file_name, iou] + a + b)
                    exclude_i.add(i)
                    exclude_j.add(j)

        for i, d in enumerate(dt):
            d = dt[i]
            if i not in exclude_i:
                a = [d["label"], d["score"]] + d["bbox"][2:]
                b = ["none", 0., 0, 0]
                vals.append([file_name, 0.] + a + b)

        for j, g in enumerate(gt):
            g = gt[j]
            if j not in exclude_j:
                a = ["none", 0., 0, 0]
                b = [g["label"], g["score"]] + g["bbox"][2:]
                vals.append([file_name, 0.] + a + b)

    names = "file_name,iou,label,score,w,h,gt_label,gt_score,gt_w,gt_h".split(",")
    data = [{a: json_type(b) for a, b in zip(names, val)} for val in vals]
    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def display_dataset(results, score_thr, output_dir, **kwargs):
    """Show model prediction results, allow gt is empty.

    Args:
        results (list): List of `(file_name, none, none, dt, none)`.
            Via `simplecv.mmdet_v2.py_test.test_dir()`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
        output_dir (str): Directory where the draft are saved.
    Returns:
        None.
    """
    simple = kwargs.get("simple", True)
    ext_file = kwargs.get("ext_file", None)
    clean_mode = kwargs.get("clean_mode", "min")
    clean_param = kwargs.get("clean_param", 0.1)
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        shutil.copy(results, output_dir)
        results = io.load_pkl(results)

    targets = None
    if ext_file is not None:
        shutil.copy(ext_file, output_dir)
        targets = set(pd.read_csv(ext_file)["file_name"].to_list())

    for file_name, _, _, dt, gt in results:
        if targets is not None and file_name not in targets:
            continue

        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if simple:
            dt = nms.clean_by_bbox(dt, clean_mode, clean_param)

        viz.image_show([os.path.join(output_dir, "images-pred")], file_name, dt, gt, None, None)
    return str(output_dir)
