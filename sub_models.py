import tensorflow.keras as tfk
import tensorflow as tf
import functools
import numpy as np


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(tfk.layers.BatchNormalization, trainable=True)
    elif norm_type == 'instance':
        NotImplementedError('Tensorflow 2.0 doesn\'t have an instance normalization implementation yet.')
        # norm_layer = tfk.layers.Insta
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


class GANLoss():
    def __init__(self, opt):
        self.target_tensor_ones = None
        self.target_tensor_zeros = None
        if opt.no_lsgan:
            self.loss = tfk.losses.BinaryCrossentropy()
        else:
            self.loss = tfk.losses.MeanSquaredError()

    def get_target_tensor(self, input, is_target_real):
        target_tensor = None
        if is_target_real:
            create_label = True if \
                ((self.target_tensor_ones is None) or (tf.size(self.target_tensor_ones) != tf.size(input))) \
                else False
            if create_label:
                self.target_tensor_ones = tf.ones_like(input)
            target_tensor = self.target_tensor_ones
        else:
            create_label = True if (
                    (self.target_tensor_zeros is None) or (tf.size(self.target_tensor_zeros) != tf.size(input))) \
                else False
            if create_label:
                self.target_tensor_zeros = tf.zeros_like(input)
            target_tensor = self.target_tensor_zeros
        return target_tensor

    def __call__(self, inputs, is_target_real):
        # print('input shape:', np.shape(inputs))
        total_loss = 0
        for input in inputs:
            # print('intermfeats:', np.shape(input[-1]))
            target_tensor = self.get_target_tensor(input[-1], is_target_real)
            total_loss += self.loss(target_tensor, input[-1])
        return total_loss


class ResnetBlock(tfk.Model):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = tfk.Sequential()
        self.conv_block.add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')))
        self.conv_block.add(tfk.layers.Conv2D(dim, kernel_size=3, padding='valid'))
        self.conv_block.add(get_norm_layer()())
        self.conv_block.add(tfk.layers.ReLU())
        self.conv_block.add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')))
        self.conv_block.add(tfk.layers.Conv2D(dim, kernel_size=3, padding='valid'))
        self.conv_block.add(get_norm_layer()())

    def call(self, inputs, training=None, mask=None):
        out = tfk.layers.add([inputs, self.conv_block(inputs)])
        return out


class GlobalGen(tfk.Model):
    def __init__(self, opt, ngf):
        super().__init__()
        self.model = tfk.Sequential()
        self.model.add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')))
        self.model.add(tfk.layers.Conv2D(ngf, kernel_size=7, padding='valid'))
        self.model.add(get_norm_layer()())
        self.model.add(tfk.layers.ReLU())

        # downsample
        for i in range(opt.n_downsample_global):
            mult = 2 ** i
            self.model.add(tfk.layers.Conv2D(ngf * mult * 2, kernel_size=3, strides=2, padding='same'))
            self.model.add(get_norm_layer()())
            self.model.add(tfk.layers.ReLU())

        # ResNet Blocks
        mult = 2 ** opt.n_downsample_global
        for i in range(opt.n_blocks_global):
            self.model.add(ResnetBlock(ngf * mult))

        # Upsample
        for i in range(opt.n_downsample_global):
            mult = 2 ** (opt.n_downsample_global - i)
            self.model.add(tfk.layers.Conv2DTranspose(int(ngf * mult / 2), kernel_size=3, strides=2, padding='same',
                                                      output_padding=1))
            self.model.add(get_norm_layer()())
            self.model.add(tfk.layers.ReLU())

        self.model.add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')))
        self.model.add(tfk.layers.Conv2D(opt.output_nc, kernel_size=7, padding='valid'))
        self.model.add(tfk.layers.Activation(tfk.activations.tanh))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class GlobalGenWithLocalEnhancers(tfk.Model):
    def __init__(self, opt):
        super().__init__()
        # for use with call():
        self.n_local_enhancers = opt.n_local_enhancers

        # Global generator layers
        ngf_global = opt.ngf * (2 ** opt.n_local_enhancers)
        self.model_global = GlobalGen(opt, ngf_global).model
        # remove last 3 layers from the global model
        for i in range(3):
            self.model_global.pop()

        # Local Enhancer layers
        self.model_downsample = dict()
        self.model_upsample = dict()

        for i in range(1, opt.n_local_enhancers + 1):
            # downsample
            ngf_global = opt.ngf * (2 ** (opt.n_local_enhancers - i))

            self.model_downsample[i] = tfk.Sequential()
            self.model_downsample[i].add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]],
                                                                            mode='REFLECT')))
            self.model_downsample[i].add(tfk.layers.Conv2D(ngf_global, kernel_size=7, padding='valid'))
            self.model_downsample[i].add(get_norm_layer()())
            self.model_downsample[i].add(tfk.layers.ReLU())
            self.model_downsample[i].add(tfk.layers.Conv2D(ngf_global * 2, kernel_size=3, strides=2, padding='same'))
            self.model_downsample[i].add(get_norm_layer()())
            self.model_downsample[i].add(tfk.layers.ReLU())

            # residual blocks
            self.model_upsample[i] = tfk.Sequential()
            for j in range(opt.n_blocks_local):
                self.model_upsample[i].add(ResnetBlock(ngf_global * 2))

            # upsample
            self.model_upsample[i].add(tfk.layers.Conv2DTranspose(ngf_global, kernel_size=3, strides=2, padding='same',
                                                                  output_padding=1))
            self.model_upsample[i].add(get_norm_layer()())
            self.model_upsample[i].add(tfk.layers.ReLU())

            # final convolution
            if i == opt.n_local_enhancers:
                self.model_upsample[i].add(tfk.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]],
                                                                              mode='REFLECT')))
                self.model_upsample[i].add(tfk.layers.Conv2D(opt.output_nc, kernel_size=7, padding='valid'))
                self.model_upsample[i].add(tfk.layers.Activation(tfk.activations.tanh))

    def call(self, inputs, training=None, mask=None):
        # print('image input shape:', np.shape(inputs))
        # create input pyramid
        input_downsampled = [inputs]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(tfk.layers.AveragePooling2D(pool_size=3, strides=2,
                                                                 padding='same')(input_downsampled[-1]))
        # print('Input shapes:')
        # for input in input_downsampled:
        #     print(input.shape, '\n', input.numpy())
        # output at coarsest level
        output_prev = self.model_global(input_downsampled[-1])

        # build up one enhancer level at a time
        for i in range(1, self.n_local_enhancers + 1):
            input_i = input_downsampled[self.n_local_enhancers - i]
            # print('i:', i, 'input_i:', input_i, 'model_downsampled[i](input_i):', self.model_downsample[i](input_i).shape,
            #       'output_prev:', output_prev.shape)
            output_prev = self.model_upsample[i](self.model_downsample[i](input_i) + output_prev)

        return output_prev


def define_G(opt):
    if opt.netG == 'global':
        netG = GlobalGen(opt, opt.ngf)
    elif opt.netG == 'global_with_local':
        netG = GlobalGenWithLocalEnhancers(opt)

    return netG


class MultiscaleDiscriminator(tfk.Model):
    def __init__(self, opt):
        super().__init__()
        self.num_D = opt.num_D
        self.n_layers_D = opt.n_layers_D
        self.getIntermFeats = not opt.no_ganFeat_loss

        self.disc = dict()
        for disc_scale in range(opt.num_D):
            self.disc[disc_scale] = self.NLayerDiscriminator(opt)

        self.downsample = tfk.layers.AveragePooling2D(pool_size=3, strides=2,
                                                      padding='same')

    def call(self, inputs, training=None, mask=None):
        all_D_results = []
        input_downsampled = inputs
        for disc_scale in range(self.num_D):
            all_D_results.append(self.disc[disc_scale](input_downsampled))
            if disc_scale != self.num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return all_D_results

    # The PatchGAN discriminator, inner class, only used by Multiscale discriminator.
    class NLayerDiscriminator(tfk.Model):
        def __init__(self, opt):
            super().__init__()
            self.getIntermFeats = not opt.no_ganFeat_loss
            self.n_layers_D = opt.n_layers_D

            kw = 4
            sequence = [[tfk.layers.Conv2D(opt.ndf, kernel_size=kw, strides=2, padding='same'),
                         tfk.layers.LeakyReLU(alpha=0.2)]]
            nf = opt.ndf
            for i in range(1, opt.n_layers_D):
                nf = min(nf * 2, 512)
                sequence += [[tfk.layers.Conv2D(nf, kernel_size=kw, strides=2, padding='same'),
                              get_norm_layer()(),
                              tfk.layers.LeakyReLU(alpha=0.2)]]

            nf = min(nf * 2, 512)
            sequence += [[tfk.layers.Conv2D(nf, kernel_size=kw, strides=1, padding='same'),
                          get_norm_layer()(),
                          tfk.layers.LeakyReLU(alpha=0.2)]]

            sequence += [[tfk.layers.Conv2D(1, kernel_size=kw, strides=1, padding='same')]]

            # Don't use Sigmoid if using LSGAN, use it if using Vanilla GAN.
            if opt.no_lsgan:
                sequence += [[tfk.layers.Activation(tfk.activations.sigmoid)]]

            # We need intermediate features from discriminator layers if using Feature Matching loss.
            if self.getIntermFeats:
                self.model_segments = dict()
                for i in range(len(sequence)):
                    self.model_segments[i] = tfk.Sequential(sequence[i])
            else:
                sequence_unfolded = []
                for i in range((len(sequence))):
                    sequence_unfolded += sequence[i]
                self.model = tfk.Sequential(sequence_unfolded)

        def call(self, inputs, training=None, mask=None):
            if self.getIntermFeats:
                results = [inputs]
                for i in range(self.n_layers_D + 2):
                    # append results of all the layers
                    results += [self.model_segments[i](results[-1])]
                # return everything except the original inputs
                return results[1:]
            else:
                # Put inside list to make consistent with above return statement when called from MultiscaleDiscriminator
                return [self.model(inputs)]


def define_D(opt):
    netD = MultiscaleDiscriminator(opt)
    return netD
