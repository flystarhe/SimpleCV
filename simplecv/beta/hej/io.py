import os

import json
import pickle
import shutil
from pathlib import Path


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
