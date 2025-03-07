import tensorflow as tf
import numpy as np
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
    sigma_max    = config.Model.sigma_max.value
    sigma_min    = config.Model.sigma_min.value
    num_scales   = config.Model.num_scales.value
    log_linspace = tf.linspace(tf.math.log(sigma_max), tf.math.log(sigma_min), num_scales)
    sigmas       = tf.exp(log_linspace) # to get the sigmas in original scale.
    return sigmas

def get_beta_schedule(type='linear'):
    """
    returns a schedule for betas for noise addition
    """
    beta_start = config.Model.sigma_max.value
    beta_end   = config.Model.sigma_min.value
    num_steps  = config.Model.num_scales.value
    if type == 'linear':
        betas = tf.linspace(beta_start, beta_end, num_steps, dtype=tf.float64)
    elif type == 'quadratic':
        betas = tf.linspace(beta_start ** 0.5 + beta_end ** 0.5, 0, num_steps, dtype=tf.float64) ** 2
    elif type == 'exponential':
        betas = tf.exp(tf.linspace(tf.math.log(beta_start), tf.math.log(beta_end), num_steps))
    elif type == 'cosine':
        betas = tf.cos(tf.linspace(0, np.pi/2, num_steps)) * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Invalid type {type} for beta schedule")
    return betas