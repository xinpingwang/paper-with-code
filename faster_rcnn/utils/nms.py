import numpy as np

from faster_rcnn.utils.iou import compute_iou


def non_max_suppression(boxes, scores, nms_threshold):
    """
    pure python nms, see the flowing link for more implementation
    # https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms

    :param boxes: bboxes location
    :param scores: score or each bbox
    :param nms_threshold:
    :return:
    """
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    # sort scores from high to low
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # put order[0] to keep and compute its iou over order[1:]
        i = order[0]
        keep.append(i)
        iou = compute_iou(boxes[i], boxes[order[1:]], areas[i], areas[order[1:]])
        ids = np.where(iou <= nms_threshold)[0] + 1
        order = order[ids]
    return np.array(keep, dtype=np.int32)
