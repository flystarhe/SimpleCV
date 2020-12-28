import copy
from pathlib import Path

from simplecv.beta.hej import io


def pad_coco_bbox(coco_file, min_size=96, pad_size=3, suffix=None):
    r'''Padding coco file with `min_size` and `pad_size`.

    Arguments:
        coco_file (str): The path of a json file with coco format
        min_size (int): If box w/h less this, will be padding
        pad_size (int): Padding box with `pad_size`
        suffix (str): Default `None`

    Examples:
        >>> coco_file = '/data/dataset1/coco.json'
        >>> pad_coco_bbox(coco_file, min_size=96, pad_size=3, suffix=None)
        >>> # Save to '/data/dataset1/coco.json.pad96_3'
    '''
    coco = io.load_coco(coco_file)

    new_annotations = []
    for ann in coco['annotations']:
        ann = copy.deepcopy(ann)
        x, y, w, h = ann['bbox']
        if 'segmentation' not in ann:
            ann['segmentation'] = None
        if w < min_size or h < min_size:
            nw = min(min_size, w + pad_size * 2) if w < min_size else w
            nh = min(min_size, h + pad_size * 2) if h < min_size else h
            ann['bbox'] = [x - (nw - w) / 2, y - (nh - h) / 2, nw, nh]
        new_annotations.append(ann)
    coco['annotations'] = new_annotations

    if suffix is None:
        suffix = '.pad{}_{}'.format(min_size, pad_size)
    out_file = Path(coco_file).with_suffix('.json' + suffix)
    io.save_json(coco, out_file)
    return out_file
