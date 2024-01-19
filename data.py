from options.train_options import TrainOptions
from glob2 import glob
import tensorflow as tf
from util import util

opt = TrainOptions().parse()
IMG_SIZE_MODULO = None


def load_and_process(data):
    for key in data:
        raw_img = tf.io.read_file(data[key])
        if (key == 'sem_map') or (key == 'inst_map'):
            data[key] = tf.io.decode_image(raw_img, dtype=tf.dtypes.uint8)
        else:
            data[key] = tf.io.decode_image(raw_img, dtype=tf.dtypes.float32)
    if 'sem_map' in data.keys():
        data['sem_map'] = tf.squeeze(data['sem_map'], [2])
        data['sem_map'] = tf.one_hot(data['sem_map'], opt.n_semantic_labels, dtype=tf.dtypes.float32)

    return data


def get_dataset(opt=opt):
    paths_dict = dict()
    if opt.use_rgb:
        paths_dict['rgb'] = sorted(glob(opt.dataroot + 'train_inp_rgb/*'))
    if opt.use_sem_map:
        paths_dict['sem_map'] = sorted(glob(opt.dataroot + 'train_sem_map/*'))
    if opt.use_inst_map:
        paths_dict['inst_map'] = sorted(glob(opt.dataroot + 'train_inst_map/*'))

    paths_dict['gt_rgb'] = sorted(glob(opt.dataroot + 'train_gt_rgb/*'))
    total_samples = len(paths_dict['gt_rgb'])
    dataset = tf.data.Dataset.from_tensor_slices(paths_dict)

    # load images and do one-hot encoding for data['sem_map']
    dataset = dataset.map(load_and_process)
    train, val = dataset.take(int(0.9 * total_samples)), dataset.skip(int(0.9 * total_samples))
    train = train.shuffle(buffer_size=900).batch(opt.batchSize)
    val = val.shuffle(buffer_size=100).repeat().batch(1)

    # For debugging:
    # count = 0
    # for epoch in range(1):
    #     for data in dataset.take(1):
    #         data = util.make_power_2(data, 32)
    #         data['edge_map'] = util.get_edge_maps(data['inst_map'])
    #         data.pop('inst_map')
    #         # plt.imshow(data['edge_map'][0, :, :, 0])
    #         # plt.show()
    #         print('Orginal semmap:\n', data['sem_map'])
    #         # print('New emap:\n', tf.dtypes.cast(data['sem_map'], tf.dtypes.float32).numpy())
    #         # print(count)
    #         # count += 1
    return train, val, total_samples


get_dataset()
