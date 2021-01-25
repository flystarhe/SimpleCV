import os

import json
import pickle
import shutil
from collections.abc import Iterable
from pathlib import Path


def to_json(v):
    try:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return {to_json(a): to_json(b) for a, b in v.items()}
        if isinstance(v, Iterable):
            return [to_json(a) for a in v]
        return float(v)
    except Exception:
        print("Unknown type:", type(v), v)
    return v


def copyfile(src, dst):
    # copies the file src to the file or directory dst
    return shutil.copy(src, dst)


def copyfile2(src, dst):
    # dst must be the complete target file name
    return shutil.copyfile(src, dst)


def load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_file):
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)
    return pkl_file


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, "w") as f:
        data = to_json(data)
        json.dump(data, f, indent=4)
    return json_file


def load_csv(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    return lines


def save_csv(lines, out_file):
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    return out_file


def load_coco(coco):
    if isinstance(coco, (str, Path)):
        coco = load_json(coco)
    return coco
