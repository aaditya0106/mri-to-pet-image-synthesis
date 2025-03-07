from utils import default_init
import tensorflow as tf
import math

class Conv3x3(tf.keras.layers.Layer):
    """
    Returns 2d conv layer with 3x3 kernel with DDPM-style initialization.
    This scaling is used to ensure that the initial activations are small and well-behaved, which can help stabilize training 
    in diffusion models. The biases are simply initialized to zero.
    """
    def __init__(self, out, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
        super(Conv3x3, self).__init__()
        padding_mode = 'same' if padding == 1 else 'valid'
        self.conv = tf.keras.layers.Conv2D(
            filters=out,
            kernel_size=3,
            strides=stride,
            padding=padding_mode,
            dilation_rate=dilation,
            use_bias=bias,
            kernel_initializer=default_init(init_scale),
            bias_initializer=tf.keras.initializers.Zeros()
        )
        
    def call(self, x):
        return self.conv(x)

class NIN(tf.keras.layers.Layer):
    """
    Network-in-Network replaces traditional 3x3 CNN with 1x1.
    Meaning each pixel in the feature map is processed independently without mixing spatial information.
    This allows the model to transform features channel-wise instead of mixing spatial locations.
    """
    def __init__(self, units, init_scale=0.1):
        super(NIN, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            units, 1, 1, padding='same', 
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=init_scale), 
            use_bias=True
        )
    
    def call(self, x):
        return self.conv(x)
    
class TimestepEmbedding(tf.keras.layers.Layer):
    """
    Implements a sinusoidal embedding for timesteps, similar to the positional embeddings used in transformers.
    Given a batch of timesteps, it returns an embedding of shape (batch_size, embedding_dim) where half of the 
    embedding dimensions use sine functions and the other half use cosine functions. 
    If embedding_dim is odd, the embedding is zero-padded.
    """
    def __init__(self, embedding_dim, max_positions=10000, **kwargs):
        super(TimestepEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

    def call(self, timesteps):
        timesteps = tf.cast(tf.reshape(timesteps, [-1]), tf.float32) # ensure timesteps is 1-D tensor of shape (batch_size,)
        half_dim = self.embedding_dim // 2
        scale = math.log(self.max_positions) / (half_dim - 1) # scaling factor: "magic number" 10000 comes from transformer paper
        exponent = tf.range(half_dim, dtype=tf.float32) * -scale # create range [0..., half_dim-1] and compute exponent weights
        emb_scale = tf.exp(exponent) # exponential decay factors
        emb = tf.expand_dims(timesteps, -1) * tf.expand_dims(emb_scale, 0) # multiplying to to get shape (batch_size, half_dim)
        emb_sin = tf.sin(emb)
        emb_cos = tf.cos(emb)
        emb = tf.concat([emb_sin, emb_cos], -1) # concat along last axis -> shape: (batch_size, 2 * half_dim)
        if self.embedding_dim % 2 == 1: # if embedding is odd, pad the last dimension with zeros
            emb = tf.pad(emb, paddings=[[0, 0], [0, 1]], mode='CONSTANT')
        emb.set_shape([None, self.embedding_dim]) # assert final shape is (batch_size, embedding_dim)
        return emb
    
class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.group_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.nin_0 = NIN(units)
        self.nin_1 = NIN(units)
        self.nin_2 = NIN(units)
        self.nin_3 = NIN(units, init_scale=0.0)

    def call(self, x):
        B, H, W, C = tf.shape(x) # batch_size, height, width, num_channels
        h = self.group_norm(x) # helps stabilize training by making training data more consistent
        q = self.nin_0(h) # query
        k = self.nin_1(h) # key
        v = self.nin_2(h) # value

        w = tf.einsum('bhwc,bijc->bhwij', q, k) * (tf.cast(C, tf.float32) ** -0.5) # scaled dot-product to find similarity between query and key
        w = tf.reshape(w, (B, H, W, H * W))
        w = tf.nn.softmax(w, axis=-1) # ensures attention scores sum to 1. higher values indicate higher importance
        w = tf.reshape(w, (B, H, W, H, W))

        h = tf.einsum('bhwij,bijc->bhwc', w, v) # apply attention weights to value
        h = self.nin_3(h)

        return x + h # residual connection

class ResnetBlockDDPM(tf.keras.layers.Layer):
    """
    The ResNet Block for diffusion models
    It applies two 3x3 convolutions with DDPM-style initialization, group normalization with 32 groups, dropout, 
    and optionally conditions the output on a time embedding.
    """
    def __init__(self, inp, out, activation, time_emb_dim=None, conv_shortcut=True, dropout=0.1):
        super(ResnetBlockDDPM, self).__init__()
        self.inp = inp
        self.out = out
        self.conv_shortcut = conv_shortcut
        self.activation = activation

        self.groupNorm0 = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.conv0 = Conv3x3(out)

        # If a time embedding dimension is provided, create a dense layer (with ddpm style initialization) to process it.
        if time_emb_dim:
            self.dense = tf.keras.layers.Dense(
                out, 
                kernel_initializer=default_init(), 
                bias_initializer=tf.keras.initializers.Zeros()
            )
        else:
            self.dense = None

        self.groupNorm1 = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.dropout0 = tf.keras.layers.Dropout(dropout)
        self.conv1 = Conv3x3(out, init_scale=1e-3) # init_scale=0 means that this layer starts with near-zero weights

        if conv_shortcut:
            self.conv2 = Conv3x3(out)
        else:
            self.nin = NIN(out)

    def call(self, x, time_emb=None, training=False):
        h = self.activation(self.groupNorm0(x, training=training))
        h = self.conv0(h)

        if time_emb is not None and self.dense is not None:
            time_emb_out = self.dense(self.activation(time_emb))
            time_emb_out = tf.reshape(time_emb_out, [-1, 1, 1, self.out])
            h = h + time_emb_out # add the time embedding bias to the features

        h = self.activation(self.groupNorm1(h, training=training))
        h = self.dropout0(h, training=training)
        h = self.conv1(h)

        if x.shape[-1] != self.out:
            x = self.conv2(x) if self.conv_shortcut else self.nin(x)

        return x + h
    
class Downsample(tf.keras.layers.Layer):
    """
    Downsamples the input by a factor of 2.
    """
    def __init__(self, out, with_conv=False):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = Conv3x3(out, stride=2, padding=0) # padding=0 means we'll handle it manually before this layer

    def call(self, x):
        _, H, W, _ = tf.shape(x)
        if self.with_conv:
            # padding="same" usually applies symmetric padding, which might not produce the same spatial alignment.
            # Therefore, we pad only right and bottom to ensure when the convolution with stride 2 is applied, 
            # the output dimensions are exactly half of the input
            pad_h = H % 2  # if height is odd, pad +1
            pad_w = W % 2  # if width is odd, pad +1
            if pad_w or pad_h:
                x = tf.pad(x, paddings=[[0, 0], [0, pad_h], [0, pad_w], [0, 0]]) # left, right, top, bottom
            x = self.conv(x)
        else:
            x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        return x

class Upsample(tf.keras.layers.Layer):
    """
    Upsamples the input by a factor of 2 using nearest neighbor interpolation.
    """
    def __init__(self, out, with_conv=False):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = Conv3x3(out)

    def call(self, x):
        _, H, W, _ = tf.shape(x)
        # Upsample spatial dimensions by a factor of 2 using nearest neighbor interpolation.
        h = tf.image.resize(x, size=(H*2, W*2), method='nearest')
        if self.with_conv:
            h = self.conv(h)
        return h