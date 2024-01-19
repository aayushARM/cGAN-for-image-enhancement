import os
import tensorflow as tf
import numpy as np


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_power_2(data, img_size_base):
    first_key = list(data.keys())[0]
    iw, ih = data[first_key].shape[1], data[first_key].shape[2]
    h = int(round(ih / img_size_base) * img_size_base)
    w = int(round(iw / img_size_base) * img_size_base)
    if h == ih and w == iw:
        return data
    for key in data:
        data[key] = tf.image.resize(data[key], (h, w), method='bilinear')

    return data


def get_edge_maps(inst_map):
    edge_map = np.zeros_like(inst_map)
    edge_map[:, :, 1:, :] = edge_map[:, :, 1:, :] | (inst_map[:, :, 1:, :] != inst_map[:, :, :-1, :])
    edge_map[:, :, :-1, :] = edge_map[:, :, :-1, :] | (inst_map[:, :, 1:, :] != inst_map[:, :, :-1, :])
    edge_map[:, 1:, :, :] = edge_map[:, 1:, :, :] | (inst_map[:, 1:, :, :] != inst_map[:, :-1, :, :])
    edge_map[:, :-1, :, :] = edge_map[:, :-1, :, :] | (inst_map[:, 1:, :, :] != inst_map[:, :-1, :, :])

    return tf.convert_to_tensor(edge_map)
