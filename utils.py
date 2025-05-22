import tensorflow as tf
import numpy as np
import config
import matplotlib.pyplot as plt

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
    sigma_max  = config.Model.sigma_max.value
    sigma_min  = config.Model.sigma_min.value
    num_scales = config.Model.num_scales.value
    t_vals     = tf.linspace(0.0, 1.0, num_scales)
    sigmas     = sigma_min * (sigma_max/sigma_min)**t_vals
    # log_linspace = tf.linspace(tf.math.log(tf.cast(sigma_max, tf.float32)), tf.math.log(tf.cast(sigma_min, tf.float32)), num_scales)
    # sigmas       = tf.exp(log_linspace) # to get the sigmas in original scale.
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
        betas = tf.exp(tf.linspace(tf.math.log(tf.cast(beta_start, tf.float32)), tf.math.log(tf.cast(beta_end, tf.float32)), num_steps))
    elif type == 'cosine':
        betas = tf.cos(tf.linspace(0, np.pi/2, num_steps)) * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Invalid type {type} for beta schedule")
    return betas

def plot_t_histogram(all_ts, step, print_every, checkpoint_dir=None, ax=None):
    """
    Plot or update a histogram of t values during training.
    If ax is provided, update the existing plot; otherwise, create a new one.
    """
    if ax is None:
        plt.figure(figsize=(6, 3))
        plt.hist(all_ts, bins=100, range=(0, 1), color="#4F8EF7", edgecolor="black", alpha=0.85)
        plt.title(f"Histogram of t values (up to step {step})", fontsize=12)
        plt.xlabel("t", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()  # Ensure plot is shown in notebook
        # if checkpoint_dir is not None:
        #     plt.savefig(os.path.join(checkpoint_dir, f"t_hist_step{step}.png"), dpi=120)
        # plt.close()
    else:
        if step == print_every:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(all_ts, bins=100, range=(0, 1), color="#4F8EF7", edgecolor="black", alpha=0.85)
            ax.set_title(f"Histogram of t values (up to step {step})", fontsize=12)
            ax.set_xlabel("t", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            ax.cla()
            ax.hist(all_ts, bins=100, range=(0, 1), color="#4F8EF7", edgecolor="black", alpha=0.85)
            ax.set_title(f"Histogram of t values (up to step {step})", fontsize=12)
            ax.set_xlabel("t", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.draw()
            plt.show()  # Ensure plot is shown in notebook
            plt.pause(0.01)
    return ax