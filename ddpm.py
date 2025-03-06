from utils import get_sigmas
import blocks
import config
import tensorflow as tf
import numpy as np

class DDPM(tf.keras.Model):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(self, activation):
        super(DDPM, self).__init__()
        self.activation_fn  = activation
        self.sigmas         = tf.constant(np.array(get_sigmas()), dtype=tf.float32)
        self.nf             = config.Model.nf # number of feature channels
        self.conditional    = config.Model.conditional
        self.scale_by_sigma = config.Model.scale_by_sigma

        # build U-Net modules
        self.time_emb_block   = blocks.TimeEmbeddingBlock(self.nf, self.activation_fn, self.conditional)
        self.downsample_block = blocks.DownsamplingBlock(self.nf, self.activation_fn)
        self.bottleneck_block = blocks.BottleneckBlock(self.nf, self.activation_fn)
        self.upsample_block   = blocks.UpsampleBlock(self.nf, self.activation_fn)
        self.final_block      = blocks.FinalBlock(self.upsample_block.out_channel, self.activation_fn)
    
    def call(self, x, labels, training=False):
        """
        Returns output of the model
        """
        time_emb = self.time_emb_block(labels) if self.conditional else None
        # process imput image. If data is already in [-1, 1], pass as is else, rescale from [0,1] to [-1,1]
        h = x if self.centered else 2*x - 1
        h, skip = self.downsample_block(h, time_emb, training=training)
        h = self.bottleneck_block(h, time_emb, training=training)
        h = self.upsample_block(h, time_emb, skip, training=training)
        h = self.final_block(h, training=training)

        if self.scale_by_sigma:
            used_sigmas = tf.gather(self.sigmas, labels) # select sigma corresponding to each label
            used_sigmas = tf.reshape(used_sigmas, [-1, 1, 1, 1])
            h = h / used_sigmas

        return h