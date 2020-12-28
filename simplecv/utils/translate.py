import numpy as np

from pathlib import Path
from simplecv.beta.hej import io


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    elif "*" in data:
        return data["*"]
    else:
        return val


def guess_code_by_image(file_name, **kwargs):
    return Path(file_name).parent.name


def guess_code_by_max_score(dt, **kwargs):
    if len(dt) == 0:
        return dict(label="__OK", score=1.0)
    return dt[np.argmax([d["score"] for d in dt])]


def guess_code_by_complex(dt, **kwargs):
    """Above the threshold are sorted by priority, the same priority are selected with higher scores.

    Args:
        dt (list): List of dict, keys `label` and `score` is required.
        code_grade (dict): Such as `dict(CODE1=LVL1,CODE2=LVL2,...)`.
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
    Returns:
        A dict, the best code.
    """
    if len(dt) == 0:
        return dict(label="__OK", score=1.0)
    code_grade, score_thr = kwargs["code_grade"], kwargs["score_thr"]
    ws = [get_val(code_grade, d["label"], 0) if d["score"] >= get_val(score_thr, d["label"], 0.3) else 0 for d in dt]
    return dt[np.argmax([w + d["score"] for w, d in zip(ws, dt)])]


def trans_test_results(results, guess_target=None, guess_predict=None, **kwargs):
    """Update `target` and `predict`, use important defect as image code.

    Args:
        results (list): List of `(file_name, none, none, dt, gt)`.
            Via `simplecv.mmdet_v1.test_net.test_coco()`.
        guess_target (function): Python funciton object, default is `None`.
        guess_predict (function): Python funciton object, default is `None`.
    Returns:
        Similar to results, list of `(file_name, target, predict, dt, gt)`.
    """
    if isinstance(results, str):
        results = io.load_pkl(results)

    if guess_target is None:
        guess_target = guess_code_by_image
    if guess_predict is None:
        guess_predict = guess_code_by_max_score
    outputs = []
    for file_name, _, _, dt, gt in results:
        predict = guess_predict(dt, **kwargs)
        target = guess_target(file_name, **kwargs)
        outputs.append([file_name, target, predict, dt, gt])
    return outputs
