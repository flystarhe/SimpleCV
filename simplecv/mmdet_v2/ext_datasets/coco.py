from mmdet.datasets import CocoDataset as _CocoDataset
from simplecv.beta.hej.io import load_json


class CocoDataset(_CocoDataset):

    def load_annotations(self, ann_file):
        cats = load_json(ann_file)["categories"]
        self.CLASSES = [cat["name"] for cat in cats]
        data_infos = super(CocoDataset, self).load_annotations(ann_file)
        return data_infos
