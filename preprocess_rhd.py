import os
import numpy as np
import cv2
import imageio
import pickle
from tqdm import tqdm


def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

def get_bbox(start, end, index, return_coords):
    visible_mask = annotations[index]['uv_vis'][start:end,2] != 0
    xs = annotations[index]['uv_vis'][start:end,0][visible_mask]
    ys = annotations[index]['uv_vis'][start:end,1][visible_mask]
    if(len(xs) > 0 and len(ys) > 0):
        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)

        width = max_x - min_x
        height = max_y - min_y

        if return_coords:
            return int(min_x), int(min_y), int(max_x), int(max_y)
        else:
            return (min_x, min_y), width, height
    else:
        return None

def get_bbox_left(index, return_coords=False):
    return get_bbox(0, 21, index, return_coords=return_coords)

def get_bbox_right(index, return_coords=False):
    return get_bbox(21, 42, index, return_coords=return_coords)

train_path = "/home/paperspace/data/rhd/RHD_published_v2/training"
mask_path = os.path.join(train_path, "mask")
color_path = os.path.join(train_path, "color")
depth_path = os.path.join(train_path, "depth")
annotations_path = os.path.join(train_path, "anno_training.pickle")

CLASS_ID = 0
FILE_NAME = "RHD_yolo_train"

with open(annotations_path, 'rb') as f:
    annotations = pickle.load(f)

with open(FILE_NAME, 'w') as w_file:
    for f in tqdm(os.listdir(color_path)):
        full_path = os.path.join(color_path, f)
        index = int(f.split('.')[0])
        im = cv2.imread(full_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        bbox_left = get_bbox_left(index, return_coords=True)
        bbox_right = get_bbox_right(index, return_coords=True)

        train_string = full_path
        if bbox_left is not None:
            train_string += f" {bbox_left[0]},{bbox_left[1]},{bbox_left[2]},{bbox_left[3]},{CLASS_ID}"
        if bbox_right is not None:
            train_string += f" {bbox_right[0]},{bbox_right[1]},{bbox_right[2]},{bbox_right[3]},{CLASS_ID}"

        w_file.write(train_string + '\n')