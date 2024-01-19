import tensorflow as tf
from options.test_options import TestOptions
import main_model
from data import get_dataset

opt = TestOptions().parse()


def test():
    model = main_model.Pix2PixHDModel(opt)
    dataset = get_dataset(opt)
    for data in dataset.take(10):
        with tf.GradientTape() as tape:
            loss_dict = model(data)
            # TODO: Put if statements to only get neccessary items from dict
            loss_D = (loss_dict['loss_D_fake'] + loss_dict['loss_D_real']) * 0.5
            loss_G = loss_dict['loss_G'] + loss_dict['feat_match_loss']

        gradients_G = tape.gradient(loss_G, model.netG.trainable_variables)
        model.optimizer_G.apply_gradients(zip(gradients_G, model.netG.trainable_variables))

        gradients_D = tape.gradient(loss_D, model.netD.trainable_variables)
        model.optimizer_D.apply_gradients(zip(gradients_D, model.netD.trainable_variables))

        print('loss_D_fake: ' + str(loss_dict['loss_D_fake'].numpy()),
              'loss_D_real: ' + str(loss_dict['loss_D_real'].numpy()),
              'loss_G: ' + str(loss_dict['loss_G'].numpy()),
              'feat_match_loss:' + str(loss_dict['feat_match_loss'].numpy()))


test()
