from mmdet.datasets import CocoDataset as _CocoDataset


class CocoDataset(_CocoDataset):

    def load_annotations(self, ann_file):
        img_infos = super(CocoDataset, self).load_annotations(ann_file)
        CocoDataset.CLASSES = tuple(cat["name"] for cat in self.coco.dataset["categories"])
        return img_infos
