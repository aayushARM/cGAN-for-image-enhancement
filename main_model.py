import tensorflow.keras as tfk
import tensorflow as tf
import sub_models
from util import util
import numpy as np


class Pix2PixHDModel(tfk.Model):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # self.input_h = None
        # self.input_w = None
        self.img_size_base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'global_with_local':
            self.img_size_base *= (2 ** opt.n_local_enhancers)

        # Create Generator
        self.netG = sub_models.define_G(opt)

        # Create Discriminator
        if opt.isTrain:
            self.netD = sub_models.define_D(opt)

        # Encoder Network:
        # Placeholder

        # load checkpoints if necessary
        if (not opt.isTrain) or opt.continue_train or opt.load_pretrain:
            pass
            # Placeholder

        if opt.isTrain:
            # Create GANLoss
            self.GANLoss = sub_models.GANLoss(opt)
            # Create feature matching loss.
            self.FeatMatchLoss = tfk.losses.MeanAbsoluteError()
            # Create optimizers
            self.optimizer_G = tfk.optimizers.Adam(opt.lr, beta_1=opt.beta1, beta_2=0.999)
            self.optimizer_D = tfk.optimizers.Adam(opt.lr, beta_1=opt.beta1, beta_2=0.999)

    def encode_inputs(self, data):
        data = util.make_power_2(data, self.img_size_base)
        if self.opt.use_inst_map:
            data['edge_map'] = util.get_edge_maps(data['inst_map'])
            data.pop('inst_map')
        g_concat_input = tf.concat([data[key] for key in data if key != 'gt_rgb'], axis=3)

        return g_concat_input, data

    def call(self, data):
        # Encode inputs
        g_concat_input, data = self.encode_inputs(data)
        # Generate fake images
        fake_images = self.netG(g_concat_input)
        # print('Generated Image:\n', fake_images.numpy())
        # Get Discriminator losses
        ## Get Discriminator predictions for Fake images
        d_concat_input_fake = tf.concat([g_concat_input, fake_images], axis=3)
        ## NOTE: Not using Pool.
        fake_pred_labels = self.netD(d_concat_input_fake)
        loss_D_fake = self.GANLoss(fake_pred_labels, is_target_real=False)

        ## Get discriminator prediction for Ground Truth images
        d_concat_input_real = tf.concat([g_concat_input, data['gt_rgb']], axis=3)
        real_pred_labels = self.netD(d_concat_input_real)
        loss_D_real = self.GANLoss(real_pred_labels, is_target_real=True)

        # Get Generator losses, note that we use self.target_tensor_ones and not self.target_tensor_zeros here.
        fake_pred_labels = self.netD(d_concat_input_fake)
        loss_G = self.GANLoss(fake_pred_labels, is_target_real=True)

        feat_match_loss = 0.0
        if not self.opt.no_ganFeat_loss:
            interm_feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            disc_weights = 1.0 / self.opt.num_D
            for disc in range(self.opt.num_D):
                for interm_feat_layer in range(self.opt.n_layers_D + 2):
                    feat_match_loss += disc_weights * interm_feat_weights \
                                       * self.FeatMatchLoss(real_pred_labels[disc][interm_feat_layer],
                                                            fake_pred_labels[disc][interm_feat_layer]) \
                                       * self.opt.lambda_feat

        # May be in future, VGG feature matching loss:
        # Placeholder

        return {'loss_D_fake': loss_D_fake, 'loss_D_real': loss_D_real, 'loss_G': loss_G,
                'feat_match_loss': feat_match_loss}

    def inference(self, data):
        g_concat_input, data = self.encode_inputs(data)
        fake_images = self.netG(g_concat_input)
        return fake_images
