import argparse
import cv2
import json
import shutil
from lxml import etree
from pathlib import Path
from xml.etree import ElementTree


ANN_EXTENSIONS = set([".xml", ".json"])
IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


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


def do_filter(img_dir, ann_dir=None):
    # `dataset name` format `xxxx_yymmdd`
    img_list = sorted(Path(img_dir).glob("**/*"))

    if ann_dir is not None:
        ann_list = sorted(Path(ann_dir).glob("**/*"))
    else:
        ann_list = img_list

    anns = {}
    for cur_file in ann_list:
        if cur_file.suffix in ANN_EXTENSIONS:
            anns[cur_file.stem] = cur_file

    imgs = {}
    for cur_file in img_list:
        if cur_file.suffix in IMG_EXTENSIONS:
            imgs[cur_file.stem] = cur_file

    ks = set(anns.keys()) & set(imgs.keys())
    data = [(imgs[k], anns[k]) for k in sorted(ks)]
    print("[abc.do_filter] cnt {}".format(len(data)))
    return data


def do_convert(img_dir, ann_dir, suffix=".jpg", color=1):
    img_dir = Path(img_dir)
    out_dir = img_dir.name + "_cvt"
    out_dir = img_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    for img_path, ann_path in do_filter(img_dir, ann_dir):
        im = cv2.imread(img_path.as_posix(), color)
        out_file = out_dir / img_path.relative_to(img_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_file.with_suffix(suffix).as_posix(), im)
        shutil.copyfile(ann_path, out_file.with_suffix(ann_path.suffix))
    print("[abc.do_convert] out {}".format(out_dir))

    xml_list = sorted(out_dir.glob("**/*.xml"))
    print("[abc.to_json] xml {}".format(len(xml_list)))
    for xml_path in xml_list:
        to_json(xml_path)
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
