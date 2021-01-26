import argparse
import cv2 as cv
import json
import pandas as pd
import shutil
from lxml import etree
from pathlib import Path
from xml.etree import ElementTree


ANN_EXTENSIONS = set([".xml", ".json"])
IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def to_json(xml_path):
    xml_path = Path(xml_path)
    parser = etree.XMLParser(encoding="utf-8")
    xmltree = ElementTree.parse(xml_path, parser=parser).getroot()

    shapes = []
    for object_iter in xmltree.findall("object"):
        name = object_iter.find("name").text
        bndbox = object_iter.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        shapes.append({"label": name, "points": [[xmin, ymin], [xmax, ymax]], "shape_type": "rectangle"})

    size = xmltree.find("size")
    imageWidth = int(float(size.find("width").text))
    imageHeight = int(float(size.find("height").text))

    data = dict(shapes=shapes, imageWidth=imageWidth, imageHeight=imageHeight)
    return save_json(data, xml_path.with_suffix(".json").as_posix())


def do_filter(img_dir, ann_dir, ext_file):
    img_list = sorted(Path(img_dir).glob("**/*"))
    img_list = [x for x in img_list if x.suffix in IMG_EXTENSIONS]

    if ext_file is not None:
        if Path(ext_file).is_dir():
            targets = [x for x in Path(ext_file).glob("**/*")]
        elif ext_file.endswith(".csv"):
            targets = pd.read_csv(ext_file)["file_name"].to_list()
        elif ext_file.endswith(".json"):
            targets = [img["file_name"] for img in load_json(ext_file)["images"]]
        else:
            raise NotImplementedError("Not Implemented file type: " + Path(ext_file).name)

        targets = set([Path(file_name).stem for file_name in targets])
        img_list = [x for x in img_list if x.stem in targets]

    imgs = {x.stem: x for x in img_list}

    if ann_dir is None:
        ann_list = img_list
    else:
        ann_list = sorted(Path(ann_dir).glob("**/*"))
        ann_list = [x for x in ann_list if x.suffix in ANN_EXTENSIONS]

    anns = {x.stem: x for x in ann_list}

    ks = set(imgs.keys()) & set(anns.keys())
    data = [(imgs[k], anns[k]) for k in sorted(ks)]
    print("[abc.do_filter.count] {}".format(len(data)))
    return data


def do_convert(img_dir, ann_dir=None, ext_file=None, suffix=".jpg", color=1):
    # ext_file (str): coco format json file path or csv file path or image/annotation dir
    img_dir = Path(img_dir)
    out_dir = img_dir.name + "_cvt"
    out_dir = img_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)

    for img_path, ann_path in do_filter(img_dir, ann_dir, ext_file):
        im = cv.imread(img_path.as_posix(), color)
        out_file = out_dir / img_path.relative_to(img_dir)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(out_file.with_suffix(suffix).as_posix(), im)

        if ann_dir is not None:
            shutil.copyfile(ann_path, out_file.with_suffix(ann_path.suffix))

    xml_list = sorted(out_dir.glob("**/*.xml"))
    for xml_path in xml_list:
        to_json(xml_path)

    print("[abc.xml_list] {}".format(len(xml_list)))
    print("[abc.out_dir] {}".format(out_dir))
    return str(out_dir)


def main(args):
    print(do_convert(args.img_dir, args.ann_dir))


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("img_dir", type=str, help="image path")
    parser.add_argument("-a", "--ann-dir", type=str, default=None)
    args = parser.parse_args()
    print(args.__dict__)
    main(args)
