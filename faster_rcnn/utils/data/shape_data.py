import numpy as np
import math
import random
import cv2

from faster_rcnn.utils.nms import non_max_suppression


def draw_shape(image, shape, dims, color):
    """
    draw shape in image
    """
    x, y, s = dims
    if shape == 'square':
        cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
    elif shape == "circle":
        cv2.circle(image, (x, y), s, color, -1)
    elif shape == "triangle":
        points = np.array([[(x, y - s),
                            (x - s / math.sin(math.radians(60)), y + s),
                            (x + s / math.sin(math.radians(60)), y + s),
                            ]], dtype=np.int32)
        cv2.fillPoly(image, points, color)
    return image


class ShapeData:

    def __init__(self, image_size, max_obj_num=4):
        """
        :param image_size: type tuple (width, height) of the image
        :param config: dict
        """
        self.image_size = image_size
        self.max_obj_num = max_obj_num
        self.shape_type = {'square': 1, 'circle': 2, 'triangle': 3}

    def random_image(self):
        """
        generate rgb image with shape objects
        :return: image, shape location and shape type
        """
        h, w = self.image_size[0], self.image_size[1]
        # initial background
        red = np.ones((h, w, 1), dtype=np.int32) * 30
        green = np.ones((h, w, 1), dtype=np.int32) * 60
        blue = np.ones((h, w, 1), dtype=np.int32) * 90
        image = np.concatenate([red, green, blue], axis=2)
        # num of objects [1, max_obj_num]
        num_obj = random.sample(range(self.max_obj_num), 1)[0] + 1
        # variables to store shapes info
        shapes = []
        ids = np.zeros((num_obj, 1))
        bboxes = np.zeros((num_obj, 4))
        dims = np.zeros((num_obj, 3), dtype=np.int32)
        for i in range(num_obj):
            # generate and record which shape to draw
            shape = random.sample(list(self.shape_type), 1)[0]
            shapes.append(shape)
            ids[i] = self.shape_type[shape]
            # generate shape size and center point location
            size = np.random.randint(h // 16, w // 8, 1)[0]  # [1/16h, 1/8w)
            # x: [1/4w, 3/4w)  y:[1/4h, 3/4h)
            x, y = np.random.randint(w // 4, w // 4 + w // 2, 1)[0], np.random.randint(h // 4, h // 4 + h // 2, 1)[0]
            dim = x, y, size
            dims[i] = dim
            bboxes[i] = np.array([x - size, y - size, x + size, y + size])
        # using nms to remove shapes that iou >= 0.01
        keep_ids = non_max_suppression(bboxes, np.arange(num_obj), 0.01)
        bboxes = bboxes[keep_ids]
        ids = ids[keep_ids]
        shapes = [shapes[i] for i in keep_ids]
        dims = dims[keep_ids]
        for j in range(bboxes.shape[0]):
            color = random.randint(1, 255)
            shape = shapes[j]
            dim = dims[j]
            image = draw_shape(image, shape, dim, color)
        return image, bboxes, ids
