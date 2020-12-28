import os
import copy
import json
import cv2 as cv
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def agent_split(data, seed=1, train_size=50):
    if len(data) * 0.8 >= train_size:
        data = sorted(data)
        np.random.seed(seed)
        np.random.shuffle(data)
        data_train = data[:train_size]
        data_val = data[train_size:]
    else:
        data_train = data
        data_val = data
    return data_train, data_val


def get_weights(coco):
    weights = defaultdict(int)
    for ann in coco["annotations"]:
        weights[ann["category_id"]] += 1
    return weights


def get_group_id(ranks):
    if len(ranks) > 0:
        ranks = sorted(ranks, key=lambda x: x[1])
        group_id = ranks[0][0]
    else:
        group_id = -1
    return group_id


def save_dataset(coco, image_ids, file_name):
    coco, image_ids = copy.deepcopy(coco), set(image_ids)
    coco["images"] = [img for img in coco["images"] if img["id"] in image_ids]
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]
    file_name.parent.mkdir(parents=True, exist_ok=True)
    save_json(coco, file_name)
    return file_name


def split_dataset(in_dir, seed=1, train_size=50):
    in_dir = Path(in_dir)

    coco = load_json(in_dir / "coco.json")

    weights = get_weights(coco)
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    data = defaultdict(list)
    for img in coco["images"]:
        ranks = []
        image_id = img["id"]
        for ann in img_anns[image_id]:
            category_id = ann["category_id"]
            ranks.append((category_id, weights[category_id]))
        data[get_group_id(ranks)].append(image_id)
    print("[selection]", [(k, len(data[k])) for k in sorted(data.keys())])

    data_train, data_val = [], []
    for _, sub_data in data.items():
        _data_train, _data_val = agent_split(sub_data, seed, train_size)
        data_train.extend(_data_train)
        data_val.extend(_data_val)
    data_test = list(set(data_train + data_val))
    save_dataset(coco, data_train, in_dir / "annotations/train.json")
    save_dataset(coco, data_test, in_dir / "annotations/test.json")
    save_dataset(coco, data_val, in_dir / "annotations/val.json")
    print("[selection] train/test/val: {}/{}/{}".format(len(data_train), len(data_test), len(data_val)))
    return data_train, data_val
