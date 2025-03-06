from utils import default_init
import layers as l
import tensorflow as tf
from functools import partial
import config

class TimeEmbeddingBlock(tf.keras.layers.Layer):
    """
    Processes timestep labels with a two-layer MLP and returns a time embedding
    """
    def __init__(self, nf, activation_fn, conditional=False):
        super(TimeEmbeddingBlock, self).__init__()
        self.conditional = conditional
        self.activation_fn = activation_fn
        if self.conditional:
            self.dense0 = tf.keras.layers.Dense(
                4*self.nf,
                kernel_initializer=default_init()((nf, 4*nf)),
                bias_initializer=tf.keras.initializers.Zeros()
            )
            self.dense1 = tf.keras.layers.Dense(
                kernel_initializer=default_init()((4*nf, 4*nf)),
                bias_initializer=tf.keras.initializers.Zeros()
            )

    def call(self, labels):
        if not self.conditional:
            return None
        time_emb = l.TimestepEmbedding(labels, self.dense0.units)
        time_emb = self.dense0(time_emb)
        time_emb = self.dense1(self.activation_fn(time_emb))
        return time_emb

class DownsamplingBlock(tf.keras.layers.Layer):
    """
    Applies an initial 3x3 convolution, then iterates over resolution levels with resnet blocks,
    optional attention, and downsampling.
    Records intermediate feature maps for skip connections.
    """
    def __init__(self, nf, activation_fn):
        super(DownsamplingBlock, self).__init__()
        self.nf = nf
        self.num_resnet_blocks = config.Model.num_res_blocks
        self.attn_resolutions  = config.Model.attn_resolutions
        self.resamp_with_conv  = config.Model.resamp_with_conv
        self.channel_mult      = config.Model.channel_mult # channel multiplier for each resolution channel
        self.num_resolutions   = len(self.channel_mult)    # number of times downsample
        self.all_resolutions   = [config.Data.image_size // (2 ** i) for i in range(len(self.channel_mult))]
        self.has_channels      = [nf] # Stores channels at each intermediate stage

        self.ResnetBlockPartial = partial(l.ResnetBlockDDPM, activation=activation_fn, time_emb_dim=4*nf, dropout=config.model.dropout)
        self.AttnBlockPartial   = partial(l.Attention)
        self.create_block()
        
    def create_block(self):
        # Initial convolution on input image
        self.layers_list  = [l.Conv3x3(self.nf)]
        in_channel = self.nf

        # loop over each resolution level
        for level in range(self.num_resolutions):
            out_channel = self.nf * self.channel_mult[level]
            self.layers_list.append(self.ResnetBlockPartial(in_channel, out_channel)) # add resnet block
            in_channel = out_channel # update number of channels
            if self.all_resolutions[level] in self.attn_resolutions:
                self.layers_list.append(self.AttnBlockPartial(out_channel)) # add an attention block
            self.has_channels.append(in_channel)

            if level < self.num_resolutions - 1:
                self.layers_list.append(l.Downsample(in_channel, with_conv=self.resamp_with_conv))
                self.has_channels.append(in_channel)

        self.out_channel = in_channel

    def call(self, x, time_emb, training=False):
        skip_connections = []
        h = x
        for layer in self.layers_list:
            try:
                h = layer(h, time_emb, training=training) # apply resnet layer
            except TypeError:
                try:
                    h = layer(h, training=training) # apply attention layer
                except Exception:
                    h = layer(h) # apply downsample layer
            skip_connections.append(h)
        return h, skip_connections
    
class BottleneckBlock(tf.keras.layers.Layer):
    """
    At the bottleneck of the U-Net, applies a sequence of residual and attention blocks.
    """
    def __init__(self, nf, activation_fn):
        super(BottleneckBlock, self).__init__()
        self.resnet1 = l.ResnetBlockDDPM(nf, nf, activation=activation_fn, time_emb_dim=4*nf, dropout=config.Model.dropout)
        self.attn    = l.Attention(nf)
        self.resnet2 = l.ResnetBlockDDPM(nf, nf, activation=activation_fn, time_emb_dim=4*nf, dropout=config.Model.dropout)

    def call(self, x, time_emb, training=False):
        h = self.resnet1(x, time_emb, training=training)
        h = self.attn(h, training=training)
        h = self.resnet2(h, time_emb, training=training)
        return h

    def call(self):
        pass

class UpsampleBlock(tf.keras.layers.Layer):
    """
    Upsamples the feature maps.
    At each resolution, concatenates the skip connection, applies residual blocks and attention, then upsamples.
    """
    def __init__(self, nf, activation_fn):
        super(UpsampleBlock, self).__init__()
        self.nf = nf
        self.num_resnet_blocks = config.Model.num_resnet_blocks
        self.attn_resolutions  = config.Model.attn_resolutions
        self.resamp_with_conv  = config.Model.resamp_with_conv
        self.channel_mult      = config.Model.channel_mult # channel multiplier for each resolution channel
        self.num_resolutions   = len(self.channel_mult)    # number of times downsample
        self.all_resolutions   = [config.Data.image_size // (2 ** i) for i in range(len(self.channel_mult))]
        self.layers_list       = []

        self.ResnetBlockPartial = partial(l.ResnetBlockDDPM, activation=activation_fn, time_emb_dim=4*nf, dropout=config.Model.dropout)
        self.AttnBlockPartial   = partial(l.Attention)

    def call(self, x, time_emb, skip_connections, training=False):
        h = x
        # reverse looping through resolution levels
        for level in reversed(range(self.num_levels)):
            for block in range(self.num_resnet_blocks + 1):
                skip = skip_connections.pop()
                h = tf.concat([h, skip], axis=-1) # concatenate skip connection from the downsampling block
                out_channel = self.nf * self.channel_mult[level]
                h = self.ResnetBlockPartial(h.shape[-1], out_channel)(h, time_emb, training=training)
            # apply attention if the current resolution is in the attn resolutions
            if self.all_resolutions[level] in self.attn_resolutions:
                h = self.AttnBlockPartial(h.shape[-1])(h, training=training)
            # upsample if not at the highest resolution.
            if level > 0:
                upsample = l.Upsample(h.shape[-1], with_conv=self.resamp_with_conv)
                h = upsample(h)
        return h

class FinalBlock(tf.keras.layers.Layer):
    """
    Applies group normalization, activation, and 3x3 convolution to produce the output image
    """
    def __init__(self, out_channels, activation_fn):
        super(FinalBlock, self).__init__()
        self.group_norm = tf.keras.layers.GroupNormalization(axis=-1, epsilon=1e-6)
        self.conv = l.Conv3x3(out_channels, init_scale=0.)
        self.activation_fn = activation_fn

    def call(self, x, training=False):
        h = self.group_norm(x, training=training)
        h = self.activation_fn(h)
        h = self.conv(h)
        return h