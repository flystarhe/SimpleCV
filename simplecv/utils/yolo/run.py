# https://github.com/ultralytics/JSON2YOLO
import sys
import json
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path


def make_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path


def save_images(ori_dir, out_dir, images):
    for img in images:
        file_name = img["file_name"]
        out_path = out_dir / Path(file_name).name
        shutil.copyfile(ori_dir / file_name, out_path)


def convert_coco_json(coco_dir, json_dir):
    coco_dir = Path(coco_dir)
    json_dir = coco_dir / json_dir
    out_dir = coco_dir.name + "_yolo"
    out_dir = coco_dir.parent / out_dir

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for json_file in sorted(json_dir.glob("*.json")):
        label_dir = out_dir / "labels" / json_file.stem
        image_dir = out_dir / "images" / json_file.stem
        make_dir(label_dir)
        make_dir(image_dir)

        with open(json_file, "r") as f:
            data = json.load(f)

        save_images(coco_dir, image_dir, data["images"])
        images = {"%g" % x["id"]: x for x in data["images"]}
        cvt_id = {c["id"]: i for i, c in enumerate(data["categories"], 0)}
        names = [c["supercategory"] + "." + c["name"] for c in data["categories"]]

        with open(out_dir / "names.txt", "a") as file:
            file.write("{}: {}".format(json_file.stem, names))

        for x in tqdm(data["annotations"], desc="Annotations %s" % json_file):
            if x.get("iscrowd"):
                continue

            img = images["%g" % x["image_id"]]
            h, w, f = img["height"], img["width"], img["file_name"]

            # format is [top left x, top left y, width, height]
            box = np.array(x["bbox"], dtype=np.float)
            box[:2] += box[2:] / 2  # to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w * h > 0
                with open(label_dir / (Path(f).stem + ".txt"), "a") as file:
                    file.write("%g %.6f %.6f %.6f %.6f\n" % (cvt_id[x["category_id"]], *box))
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _, coco_dir, json_dir = sys.argv
        print(convert_coco_json(coco_dir, json_dir))
    else:
        print("Format: `python run.py coco/dir/ json/dir/`")
