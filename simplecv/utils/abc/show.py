import argparse
import cv2 as cv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def draw_bbox(anns, img_path, color_val):
    img = cv.imread(str(img_path), 1)
    if not isinstance(anns, list):
        return img

    img_h, img_w = img.shape[:2]
    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv.rectangle(img, (x, y), (x + w, y + h), color_val, thickness=2)

        if y > 30:
            left_bottom = (x, y)
        elif h > img_h * 0.5:
            left_bottom = (x, y + h)
        else:
            left_bottom = (x, y + h + 30)

        text = "{}: {}/{}={:.2f}".format(ann["label"], h, w, h / w)
        cv.putText(img, text, left_bottom, cv.FONT_HERSHEY_COMPLEX, 1.0, color_val)

    cv.putText(img, img_path.parent.name, (30, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, color_val)
    return img


def show_dataset(img_dir, out_dir):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    coco = load_json(img_dir / "coco.json")

    id2label = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img2anns = defaultdict(list)
    for ann in coco["annotations"]:
        ann["label"] = id2label[ann["category_id"]]
        img2anns[ann["image_id"]].append(ann)

    gts, imgs = [], []
    for img in coco["images"]:
        gts.append(img2anns[img["id"]])
        imgs.append(img_dir / img["file_name"])

    for gt, img_path in zip(gts, imgs):
        img = draw_bbox(gt, img_path, (0, 0, 255))
        cv.imwrite(str(out_dir / img_path.name), img)


def main(args):
    print(show_dataset(args.img_dir, args.out_dir))


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("img_dir", type=str, help="image dir")
    parser.add_argument("out_dir", type=str, help="output dir")
    args = parser.parse_args()
    print(args.__dict__)
    main(args)
