import json
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


def load_coco(coco):
    if isinstance(coco, (str, Path)):
        with open(coco, "r") as f:
            coco = json.load(f)
    return coco


def show_line(title, values, labels, fmt="{}:{}"):
    lst = [fmt.format(l, v) for v, l in zip(values, labels)]
    print("{} -> {}".format(title, ", ".join(lst)))


def norm(x):
    if x >= 1.0:
        return int(x)
    else:
        return int(1.0 / (x + 1e-5))


def grid(xs, ys):
    xs = [20 if x > 20 else x for x in xs]
    ys = [20 if y > 20 else y for y in ys]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    shape = (y_max - y_min + 1, x_max - x_min + 1)
    data = np.zeros(shape, dtype="int16")
    for x, y in zip(xs, ys):
        i, j = y - y_min, x - x_min
        data[i, j] = data[i, j] + 1
    row_labels = list(range(y_min, y_max + 1))
    col_labels = list(range(x_min, x_max + 1))
    return data, row_labels, col_labels


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_scale(anns):
    boxes = np.array([a["bbox"] for a in anns])
    scales = np.sqrt(np.prod(boxes[:, 2:4], axis=1))
    h_ratios = [h / s for h, s in zip(boxes[:, 3], scales)]
    h_ratios_ = np.square(h_ratios)
    scales_ = np.log2(scales)

    print("\nscales-vs-ratios")
    display(scales_, h_ratios_)

    x_tmp = [1 for _ in scales_]
    print("\nall-vs-scales")
    display(x_tmp, scales_)

    x_tmp = [1 for _ in h_ratios_]
    print("\nall-vs-ratios")
    display(x_tmp, h_ratios_)

    vals = sorted(set([a["label"] for a in anns]))
    mapping = {v: i for i, v in enumerate(vals, 1)}
    h_ratios_ = [mapping[a["label"]] for a in anns]

    print("\nscales-vs-categories", json.dumps(mapping, sort_keys=True))
    display(scales_, h_ratios_)

    x_tmp = [1 for _ in h_ratios_]
    print("\nall-vs-categories", json.dumps(mapping, sort_keys=True))
    display(x_tmp, h_ratios_)


def display(scales_, h_ratios_):
    scales_ = [norm(s) for s in scales_]
    h_ratios_ = [norm(r) for r in h_ratios_]
    data, row_labels, col_labels = grid(h_ratios_, scales_)

    fig, ax = plt.subplots(figsize=(16, 6))
    im, cbar = heatmap(data, row_labels, col_labels, ax=ax, cmap="YlGn", cbarlabel="counts")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")
    fig.tight_layout()
    plt.show()

    show_line("ROW", data.sum(axis=1), row_labels)
    show_line("COL", data.sum(axis=0), col_labels)
    print("Total:", data.sum())


def do_analyze(coco_file):
    coco = load_coco(coco_file)

    id2label = {}
    for cat in coco["categories"]:
        id2label[cat["id"]] = cat["name"]

    anns = []
    for ann in coco["annotations"]:
        ann["label"] = id2label[ann["category_id"]]
        anns.append(ann)

    return plot_scale(anns)
