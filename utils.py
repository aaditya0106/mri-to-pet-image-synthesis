import tensorflow as tf
import config

def scaled_initializer(init_scale, base_initializer=tf.keras.initializers.GlorotUniform()):
    """
    Returns an initializer that scales the base initializer by init_scale. If init_scale is 0, it uses a very small number (1e-10).
    """
    def initializer(shape, dtype=None):
        scale = 1e-10 if init_scale == 0 else init_scale
        return base_initializer(shape, dtype=dtype) * scale
    return initializer

def default_init(init_scale=1.):
    """
    Returns an initializer that scales weights by init_scale.
    """
    return tf.keras.initializers.VarianceScaling(scale=init_scale, mode='fan_avg', distribution='uniform')

def get_sigmas():
    """
    Returns an arrary of noise levels
    """
    sigma_max    = config.Model.sigma_max
    sigma_min    = config.Model.sigma_min
    num_scales   = config.Model.num_scales
    log_linspace = tf.linspace(tf.math.log(sigma_max), tf.math.log(sigma_min), num_scales)
    sigmas       = tf.exp(log_linspace) # to get the sigmas in original scale.
    return sigmas