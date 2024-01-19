import tensorflow as tf
from options.train_options import TrainOptions
import main_model
from util import util
from data import get_dataset
import numpy as np
from PIL import Image
from math import ceil

opt = TrainOptions().parse()


def train():
    model = main_model.Pix2PixHDModel(opt)
    train_data, val_data, num_samples = get_dataset(opt)
    img_save_freq = 400
    util.mkdir(opt.checkpoints_dir + '/' + opt.name + '/gen_images')
    # Run once to generate weights.
    model(next(iter(train_data)))
    var_dict = dict()
    for var in model.trainable_weights:
        var_dict[var.name] = var
    model_ckpt = tf.train.Checkpoint(**var_dict)
    optimizers_ckpt = tf.train.Checkpoint(optimizer_G=model.optimizer_G, optimizer_D=model.optimizer_D)
    start_epoch = 1

    if opt.continue_train:
        model_ckpt.restore(tf.train.latest_checkpoint(opt.checkpoints_dir + '/' + opt.name + '/model_ckpts'))
        optimizers_ckpt.restore(tf.train.latest_checkpoint(opt.checkpoints_dir + '/' + opt.name + '/optimizer_ckpts'))
        with open(opt.checkpoints_dir + '/' + opt.name + '/last_epoch.txt', 'r') as ckpt_txt_file:
            start_epoch = int(ckpt_txt_file.read())

    # with tf.device('/gpu:1'):
    for epoch in range(start_epoch, opt.n_epoch + opt.n_epoch_decay + 1):
        for train_batch, val_batch, batch_num in zip(train_data, val_data,
                                                     range(1, ceil(num_samples / opt.batchSize) + 1)):
            with tf.GradientTape(persistent=True) as tape:
                loss_dict = model(train_batch)
                # loss_dict = func(train_batch)
                # TODO: Put if statements to only get neccessary items from dict
                loss_D = (loss_dict['loss_D_fake'] + loss_dict['loss_D_real']) * 0.5
                loss_G = loss_dict['loss_G'] + loss_dict['feat_match_loss']

            gradients_G = tape.gradient(loss_G, model.netG.trainable_variables)
            model.optimizer_G.apply_gradients(zip(gradients_G, model.netG.trainable_variables))

            gradients_D = tape.gradient(loss_D, model.netD.trainable_variables)
            model.optimizer_D.apply_gradients(zip(gradients_D, model.netD.trainable_variables))

            print('[epoch %d, batch %d]-> ' % (epoch, batch_num) + 'loss_D_fake: ' + str(
                loss_dict['loss_D_fake'].numpy()),
                  'loss_D_real: ' + str(loss_dict['loss_D_real'].numpy()),
                  'loss_G: ' + str(loss_dict['loss_G'].numpy()),
                  'feat_match_loss:' + str(loss_dict['feat_match_loss'].numpy()))

            if batch_num % img_save_freq == 0:
                fake_image = model.inference(val_batch)[0]
                with open(opt.checkpoints_dir + '/' + opt.name + '/interm_float_imgs.txt', 'a+') as img_txt_file:
                    img_txt_file.write('Float image at epoch %d, batch %d:\n' % (epoch, batch_num))
                    img_txt_file.write(str(fake_image.numpy()) + '\n\n')
                fake_image = tf.image.convert_image_dtype(fake_image, tf.uint8).numpy()
                im = Image.fromarray(fake_image)
                im.save(opt.checkpoints_dir + '/' + opt.name + '/gen_images/' + 'img_epoch%d_iter%d.png' % (
                    epoch, batch_num))
            # End batch.

        if epoch % opt.save_epoch_freq == 0:
            model_ckpt.save(opt.checkpoints_dir + '/' + opt.name + '/model_ckpts/ckpt_epoch_%d' % (epoch))
            optimizers_ckpt.save(opt.checkpoints_dir + '/' + opt.name + '/optimizer_ckpts/ckpt_epoch_%d' % (epoch))
            with open(opt.checkpoints_dir + '/' + opt.name + '/last_epoch.txt', 'w+') as ckpt_txt_file:
                ckpt_txt_file.write(str(epoch))

        if epoch > opt.n_epoch:
            lr = model.optimizer_G.learning_rate.numpy()  # or optimizer_D, both are same
            lr = lr - (lr / opt.n_epoch_decay)  # linear decay
            model.optimizer_G.learning_rate = tf.constant(lr, dtype=tf.dtypes.float32)
            model.optimizer_D.learning_rate = tf.constant(lr, dtype=tf.dtypes.float32)
            print('Updated learning rate to:', model.optimizer_G.learning_rate.numpy())
        # End epoch


train()
