import argparse
import multiprocessing
import os
import os.path as osp
import subprocess
import time

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from simplecv.beta.hej import io
from simplecv.utils.analyze import matrix_analysis_image
from simplecv.utils.analyze import matrix_analysis_object


IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


G_THIS_DIR = osp.dirname(__file__)
G_COMMAND = "CUDA_VISIBLE_DEVICES={} PYTHONPATH={} python {}/py_app.py {} {} {} {}"
G_PYTHONPATH = "{}:{}".format(os.environ["SIMPLECV_PATH"], os.environ["MMDET_PATH"])


def system_command(params):
    gpu_id, file_list, argv = params
    # argv: (config, checkpoint, patch_size)

    logs = []
    for file_name in file_list:
        command_line = G_COMMAND.format(gpu_id, G_PYTHONPATH, G_THIS_DIR, file_name, *argv)
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            logs.append("GPU{} {}: {}".format(gpu_id, file_name, result.args))
        else:
            logs.append("GPU{} {}: {}".format(gpu_id, file_name, "OK."))
    return "\n".join(logs)


def collect_results(pkl_list):
    results = []
    for i, pkl_file in enumerate(pkl_list):
        if osp.isfile(pkl_file):
            data = io.load_pkl(pkl_file)
            results.extend(data)
        else:
            print("Failed task[{}]: {}".format(i, pkl_file))
    return results


def multi_gpu_test(dataset, config, checkpoint, patch_size, gpus=4):

    def task_split(dataset, splits, tmp_dir):
        file_list = []
        n = len(dataset)
        block_size = (n - 1) // splits + 1
        os.makedirs(tmp_dir, exist_ok=True)
        for i, i_start in enumerate(range(0, n, block_size)):
            out_file = osp.join(tmp_dir, "part{:02d}".format(i))
            subset = dataset[i_start: i_start + block_size]
            io.save_json(subset, out_file)
            file_list.append(out_file)
        return file_list

    argv = (config, checkpoint, patch_size)
    in_files = task_split(dataset, gpus * 4, "tmp/")
    command_params = [(i, in_files[i::gpus], argv) for i in range(gpus)]

    pool = multiprocessing.Pool(processes=gpus)
    logs = pool.map(system_command, command_params)
    print("multi-gpu test logs:\n" + "\n".join(logs))
    pkl_list = [in_file + ".out" for in_file in in_files]
    return collect_results(pkl_list)


def xywh2xyxy(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] + _bbox[0],
        _bbox[3] + _bbox[1],
    ]


def test_dir(data_root, config, checkpoint, patch_size, gpus=4):
    """Test model with multiple gpus.

    Args:
        data_root (str): Dataset root dir.
        config (str): Path of config file full path.
        checkpoint (str): Path of checkpoint file full path.
        gpus (int, tuple): The number of GPUs or list of available GPU.
    Returns:
        The test results, list of `(file_name, target, predict, dt, gt)`.
    """
    temp_file = osp.join(osp.dirname(checkpoint), "mmdet_v2_test_{}.pkl".format(time.strftime("%m%d%H%M")))
    imgs = [img.as_posix() for img in Path(data_root).glob("**/*.*") if img.suffix in IMG_EXTENSIONS]

    results = multi_gpu_test(imgs, config, checkpoint, patch_size, gpus)
    assert len(imgs) == len(results)

    outputs = []
    for file_name, dt in results:
        outputs.append((file_name, None, None, dt, []))
    return io.save_pkl(outputs, temp_file)


def test_coco(data_root, coco_file, config, checkpoint, patch_size, gpus=4):
    """Test model with multiple gpus.

    Args:
        data_root (str): Dataset root dir.
        coco_file (str): Coco subset json file.
        config (str): Path of config file full path.
        checkpoint (str): Path of checkpoint file full path.
        gpus (int, tuple): The number of GPUs or list of available GPU.
    Returns:
        The test results, list of `(file_name, target, predict, dt, gt)`.
    """
    temp_file = osp.join(osp.dirname(checkpoint), "mmdet_v2_test_{}.pkl".format(time.strftime("%m%d%H%M")))

    coco_file = osp.join(data_root, coco_file)
    coco = io.load_json(coco_file)

    id2label = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img2anns = defaultdict(list)
    for ann in coco["annotations"]:
        ann["label"] = id2label[ann["category_id"]]
        ann["xyxy"] = xywh2xyxy(ann["bbox"])
        ann["score"] = 1.0
        img2anns[ann["image_id"]].append(ann)

    gts, imgs = [], []
    for img in coco["images"]:
        gts.append(img2anns[img["id"]])
        imgs.append(osp.join(data_root, img["file_name"]))

    results = multi_gpu_test(imgs, config, checkpoint, patch_size, gpus)
    assert len(imgs) == len(results)

    outputs = []
    for gt, (file_name, dt) in zip(gts, results):
        outputs.append((file_name, None, None, dt, gt))
    kwargs = dict(clean_mode="min", clean_param=0.1, match_mode="iou", pos_iou_thr=0.1, min_pos_iou=0.01)
    matrix_analysis_object(outputs, {"*": 0.5}, temp_file + ".object.csv", **kwargs)
    return io.save_pkl(outputs, temp_file)


def main(args):
    if args.coco == "none":
        return test_dir(args.root, args.config, args.checkpoint, args.patch_size, args.gpus)
    return test_coco(args.root, args.coco, args.config, args.checkpoint, args.patch_size, args.gpus)


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("root", type=str, help="dataset root")
    parser.add_argument("coco", type=str, help="coco json file")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("checkpoint", type=str, help="checkpoint file path")
    parser.add_argument("patch_size", type=int, default=999999, help="patch size")
    parser.add_argument("--gpus", type=int, default=4, help="number of gpus to use")
    args = parser.parse_args()
    print(main(args))
