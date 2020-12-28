import glob
import re
from pathlib import Path


def make_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path


def increment_path(path, exist_ok=True, sep=""):
    path = Path(path)

    if not path.exists() or exist_ok:
        return make_dir(path)

    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 2  # increment number
    return make_dir(f"{path}{sep}{n}")
