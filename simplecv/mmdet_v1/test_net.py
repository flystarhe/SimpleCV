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


G_THIS_DIR = osp.dirname(__file__)
G_COMMAND = "CUDA_VISIBLE_DEVICES={} PYTHONPATH={} python {}/test.py {} {} {}"
G_PYTHONPATH = '{}:{}'.format(os.environ["SIMPLECV_PATH"], os.environ['MMDET_PATH'])


def system_command(command_line):
    result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout.decode("utf8").strip()
    return ""


def collect_results(pkl_list):
    results = []
    code_names = []
    for i, pkl_file in enumerate(pkl_list):
        if osp.isfile(pkl_file):
            data = io.load_pkl(pkl_file)
            results.extend(data[0])
            code_names = data[1]
        else:
            print("Failed task[{}]: {}".format(i, pkl_file))
    return results, code_names


def multi_gpu_test(dataset, config, checkpoint, gpus=4):
    def task_split(dataset, splits, tmp_dir):
        file_list = []
        n = len(dataset)
        block_size = (n - 1) // splits + 1
        os.makedirs(tmp_dir, exist_ok=True)
        for i, i_start in enumerate(range(0, n, block_size)):
            out_file = osp.join(tmp_dir, "part{:02d}".format(i))
            subset = dataset[i_start:i_start + block_size]
            io.save_json(subset, out_file)
            file_list.append(out_file)
        return file_list

    if isinstance(gpus, int):
        gpus = list(range(gpus))
    file_list = task_split(dataset, len(gpus), "tmp")
    pool = multiprocessing.Pool(processes=len(file_list))

    command_list = []
    for i, filename in zip(gpus, file_list):
        command_list.append(G_COMMAND.format(i, G_PYTHONPATH, G_THIS_DIR, filename, config, checkpoint))

    results = pool.map(system_command, command_list)
    results = [r.split("\n")[-1] for r in results]
    return collect_results(results)


def xyxy2xywh(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def bboxes2anns(bbox_, code_):
    anns = []
    for i in range(bbox_.shape[0]):
        xyxys = bbox_[i].tolist()
        x, y, w, h = xyxy2xywh(xyxys)
        ann = dict(label=code_, bbox=[x, y, w, h], xyxy=xyxys[:4], score=xyxys[4], area=(w * h))
        anns.append(ann)
    return anns


def test_dir(data_root, config, checkpoint, gpus=4):
    """Test model with multiple gpus.

    Args:
        data_root (str): Dataset root dir.
        config (str): Path of config file full path.
        checkpoint (str): Path of checkpoint file full path.
        gpus (int, tuple): The number of GPUs or list of available GPU.
    Returns:
        The test results, list of `(file_name, target, predict, dt, gt)`.
    """
    temp_file = osp.join(osp.dirname(checkpoint), "mmdet_v1_test_{}.pkl".format(time.strftime("%m%d%H%M")))
    imgs = [img.as_posix() for img in Path(data_root).glob("**/*.*") if img.suffix == ".jpg"]

    results, code_names = multi_gpu_test(imgs, config, checkpoint, gpus)
    assert len(imgs) == len(results)

    outputs = []
    for file_name, bbox_result, _ in results:
        dt = []
        for bbox_, code_ in zip(bbox_result, code_names):
            dt.extend(bboxes2anns(bbox_, code_))
        outputs.append((file_name, None, None, dt, None))
    return io.save_pkl(outputs, temp_file)


def test_coco(data_root, coco_file, config, checkpoint, gpus=4):
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
    temp_file = osp.join(osp.dirname(checkpoint), "mmdet_v1_test_{}.pkl".format(time.strftime("%m%d%H%M")))

    coco_file = osp.join(data_root, coco_file)
    coco = io.load_json(coco_file)

    id2label = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img2anns = defaultdict(list)
    for ann in coco["annotations"]:
        ann["label"] = id2label[ann["category_id"]]
        img2anns[ann["image_id"]].append(ann)

    gts, imgs = [], []
    for img in coco["images"]:
        gts.append(img2anns[img["id"]])
        imgs.append(osp.join(data_root, img["file_name"]))

    results, code_names = multi_gpu_test(imgs, config, checkpoint, gpus)
    assert len(imgs) == len(results)

    outputs = []
    for gt, (file_name, bbox_result, _) in zip(gts, results):
        dt = []
        for bbox_, code_ in zip(bbox_result, code_names):
            dt.extend(bboxes2anns(bbox_, code_))
        outputs.append((file_name, None, None, dt, gt))
    return io.save_pkl(outputs, temp_file)


def main(args):
    if args.coco == "none":
        return test_dir(args.root, args.config, args.checkpoint, args.gpus)
    return test_coco(args.root, args.coco, args.config, args.checkpoint, args.gpus)


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("root", type=str, help="dataset root")
    parser.add_argument("coco", type=str, help="coco json file")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("checkpoint", type=str, help="checkpoint file path")
    parser.add_argument("--gpus", type=int, default=4, help="number of gpus to use")
    args = parser.parse_args()
    print(main(args))
