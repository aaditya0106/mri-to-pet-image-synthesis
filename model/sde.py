from utils import get_beta_schedule
import config
import tensorflow as tf
import numpy as np

class VESDE(tf.keras.Model):
    """
    Variance Exploding Stochastic Differential Equation based Diffusion Model
    """
    def __init__(self, score_func):
        super(VESDE, self).__init__()
        self.T = 1.0 # end time of SDE
        self.N = config.Model.num_scales.value
        self.sigma_min = config.Model.sigma_min.value
        self.sigma_max = config.Model.sigma_max.value
        self.sigmas = get_beta_schedule('exponential')
        self.score_func = score_func

    def marginal_probability(self, x, t):
        """
        Computes the standard deviation and mean of the marginal probability distribution.
        The variance is computed exponentially using sigma_min and sigma_max.
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x # mean remains unchanged
        return mean, std
    
    def prior_sampling(self, shape):
        """
        Samples from the prior distribution, which is an isotropic Gaussian.
        Return x_T (pure noise) which is starting step of the reverse SDE.
        """
        return tf.random.normal(shape) * self.sigma_max
    
    def fwd_discrete(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + g_i z_i

        Where:
            f_i(x_i) = 0 (drift term)
            g_i = sigma_min * (sigma_max / sigma_min)^t is the diffusion coefficient

        Conceptual update (for any noise schedule):
            x_{t+1} = x_t + sqrt(sigma_{t+1}**2 - sigma_t**2) * z_t

        Practical approximation with a geometric schedule:
            x_{t+1} = x_t + (sigma_min * (sigma_max / sigma_min)**t) * z_t
        """
        _, g = self.marginal_probability(x, t)
        z = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        x = x + g * z
        return x

    def reverse_discrete(self, x, t, mri):
        """
        Discretizes the reverse SDE for stable numerical integration.
        x_i = x_{i+1} - g_{i+1}^2 s_theta(x_{i+1}, y, i+1) + g_{i+1} z_{i+1}
        Where:
            s_theta(x, y, t) is the score function
            g_{i+1} = sqrt(sigma_{i+1}^2 - sigma_i^2) is the diffusion coefficient
        """
        timestep = tf.cast(t * (self.N - 1) / self.T, tf.int64)
        sigma = tf.gather(self.sigmas, timestep)
        prev_sigma = tf.gather(self.sigmas, tf.maximum(timestep - 1, 0))

        g = tf.sqrt(sigma ** 2 - prev_sigma ** 2) # compute diffusion coefficient

        x_concat = tf.concat([x, mri], axis=-1)
        score = self.pet_score_func(x_concat, t) # compute PET score function gradient

        z = tf.random.normal(tf.shape(x), dtype=x.dtype)
        x_mean = x + g ** 2 * score
        x = x_mean + g * z
        return x, x_mean