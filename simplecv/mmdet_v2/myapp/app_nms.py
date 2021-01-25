import copy
import torch
import numpy as np
import networkx as nx
from collections import defaultdict


def clustering(nodes, lines):
    # G.add_nodes_from([1, 2, 3, 4])
    # G.add_edges_from([(1, 2), (1, 3)])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(lines)
    return list(nx.connected_components(G))


def bbox_overlaps(dt, gt=None, mode="iou"):
    # shape (n, 4) in <x1, y1, x2, y2> format.
    bboxes = [d["xyxy"] for d in dt]
    bboxes1 = torch.FloatTensor(bboxes)
    if gt is not None:
        bboxes = [g["xyxy"] for g in gt]
    bboxes2 = torch.FloatTensor(bboxes)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return None

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if "iou" == mode:
        ious = overlap / (area1[:, None] + area2 - overlap + 1.0)
    elif "min" == mode:
        ious = overlap / (torch.min(area1[:, None], area2) + 1.0)
    elif "/dt" == mode:
        ious = overlap / (area1[:, None] + 1.0)
    elif "/gt" == mode:
        ious = overlap / (area2[None, :] + 1.0)

    return ious.numpy()


def _clean_with_iou(dt, mode="min", thr=0.1):
    ious = bbox_overlaps(dt, None, mode)

    if ious is None:
        return dt

    nodes = list(range(ious.shape[0]))
    lines = np.argwhere(ious >= thr).tolist()

    dt_ = []
    for i_set in clustering(nodes, lines):
        vals = [dt[i] for i in i_set]
        best = max(vals, key=lambda x: x["score"])
        bboxes = np.array([d["xyxy"] for d in vals], dtype=np.float32)
        (x1, y1), (x2, y2) = bboxes[:, :2].min(axis=0), bboxes[:, 2:].max(axis=0)
        best["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        best["area"] = (x2 - x1) * (y2 - y1)
        best["xyxy"] = [x1, y1, x2, y2]
        dt_.append(best)
    return dt_


def _clean_with_dist(dt, k=1.5):
    # shape (n, 4) in <x1, y1, x2, y2> format.
    bboxes = [d["xyxy"] for d in dt]
    bboxes = torch.FloatTensor(bboxes)

    if bboxes.size(0) == 0:
        return dt

    xy = (bboxes[:, 2:] + bboxes[:, :2]) * 0.5
    wh = (bboxes[:, 2:] - bboxes[:, :2]).clamp(min=0)

    dist = (xy[:, None] - xy).abs()  # (n, n, 2)
    limit = torch.max(wh[:, None], wh) * k  # (n, n, 2)
    mask = torch.prod((dist <= limit).to(torch.int), -1)

    nodes = list(range(mask.size(0)))
    lines = mask.nonzero(as_tuple=False).tolist()

    dt_ = []
    for i_set in clustering(nodes, lines):
        vals = [dt[i] for i in i_set]
        best = max(vals, key=lambda x: x["score"])
        bboxes = np.array([d["xyxy"] for d in vals], dtype=np.float32)
        (x1, y1), (x2, y2) = bboxes[:, :2].min(axis=0), bboxes[:, 2:].max(axis=0)
        best["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        best["area"] = (x2 - x1) * (y2 - y1)
        best["xyxy"] = [x1, y1, x2, y2]
        dt_.append(best)
    return dt_


def clean_by_bbox(dt, mode="min", param=0.1):
    cache = defaultdict(list)
    dt = copy.deepcopy(dt)

    for d in dt:
        _, _, w, h = d["bbox"]
        if w * h >= 64:
            cache[d["label"]].append(d)

    dt_ = []
    for v in cache.values():
        if mode == "dist":
            dt_.extend(_clean_with_dist(v, param))
        else:
            dt_.extend(_clean_with_iou(v, mode, param))
    return dt_
