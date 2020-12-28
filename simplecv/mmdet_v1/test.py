import argparse
from pathlib import Path

from mmdet.apis import init_detector, inference_detector

from simplecv.beta.hej import io


################################################################
# Based on `mmdetection/mmdet/apis/inference.py`, insert code:
from mmdet.datasets.registry import DATASETS
from mmdet.datasets.registry import PIPELINES
from simplecv.mmdet_v1.ext_datasets import CocoDataset

DATASETS.register_module(CocoDataset, force=True)
################################################################


def test_imgs(img_list, config, checkpoint):
    model = init_detector(config, checkpoint)
    code_names = model.CLASSES

    results = []
    for img_path in img_list:
        img_path = Path(img_path)
        file_name = img_path.as_posix()
        result = inference_detector(model, file_name)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        results.append([file_name, bbox_result, segm_result])
    return results, code_names


def main(args):
    in_file = args.data
    img_list = io.load_json(in_file)
    outputs = test_imgs(img_list, args.config, args.checkpoint)
    return io.save_pkl(outputs, in_file + ".out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("data", type=str, help="json file path")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("checkpoint", type=str, help="checkpoint file path")
    args = parser.parse_args()
    print(main(args))
