import os

import argparse
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from simplecv.beta.hej import io


################################################################
# Based on `mmdetection/mmdet/apis/inference.py`, insert code:
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import PIPELINES
from simplecv.mmdet_v2.ext_datasets import CocoDataset
from simplecv.mmdet_v2.ext_pipelines import RandomCrop, Resize2

DATASETS.register_module(name='CocoDataset', force=True, module=CocoDataset)
PIPELINES.register_module(name='RandomCrop', force=True, module=RandomCrop)
PIPELINES.register_module(name='Resize2', force=True, module=Resize2)
################################################################


def test_imgs(img_list, config, checkpoint):
    config = Config.fromfile(config)
    config.merge_from_dict(eval(os.environ.get("CFG_OPTIONS", "{}")))
    model = init_detector(config, checkpoint)
    code_names = model.CLASSES

    results = []
    for file_name in img_list:
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
