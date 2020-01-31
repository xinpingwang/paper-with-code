import numpy as np


def compute_iou(box, boxes, area, areas):
    """
    compute ious of box over boxes
    :param box:
    :param boxes:
    :param area: area of box
    :param areas: areas of target boxes
    :return: ious
    """
    # max of top left and min of bottom down
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    overlapping = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
    union = areas[:] + area - overlapping
    iou = overlapping / union
    return iou
