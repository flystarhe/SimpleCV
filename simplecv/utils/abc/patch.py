import argparse
import cv2 as cv
import shutil
from pathlib import Path


IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def split(size, patch_size, overlap=128):
    if patch_size >= size:
        return [0]

    s = list(range(0, size - patch_size, patch_size - overlap))
    s.append(size - patch_size)
    return s


def _patch_img(out_dir, img_path, patch_size, color_mode=1):
    img = cv.imread(str(img_path), color_mode)
    img_h, img_w = img.shape[:2]

    ys = split(img_h, patch_size)
    xs = split(img_w, patch_size)

    cnt = 1
    stem = img_path.stem
    suffix = img_path.suffix
    for y in ys:
        for x in xs:
            out_file = out_dir / "{}-{:02d}{}".format(stem, cnt, suffix)
            sub_img = img[y: y + patch_size, x: x + patch_size]
            cv.imwrite(str(out_file), sub_img)
            cnt += 1


def do_patch(img_dir, out_dir, patch_size, color_mode=1):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    imgs = [img for img in img_dir.glob("**/*") if img.suffix in IMG_EXTENSIONS]
    print("[patch.do_patch] imgs: {}".format(len(imgs)))

    for img_path in imgs:
        _patch_img(out_dir, img_path, patch_size, color_mode)
    return str(out_dir)


def main(args):
    print(do_patch(args.img_dir, args.out_dir, args.patch_size))


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("img_dir", type=str, help="image dir")
    parser.add_argument("out_dir", type=str, help="output dir")
    parser.add_argument("patch_size", type=int, help="patch size")
    args = parser.parse_args()
    print(args.__dict__)
    main(args)
