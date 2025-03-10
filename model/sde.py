from utils import get_beta_schedule
import config
import tensorflow as tf
import numpy as np

class VESDE(tf.keras.Model):
    """
    Variance Exploding Stochastic Differential Equation based Diffusion Model
    """
    def __init__(self, pet_score_func, mri_score_func):
        super(VESDE, self).__init__()
        self.T = 1.0 # end time of SDE
        self.N = config.Model.num_scales.value
        self.sigma_min = config.Model.sigma_min.value
        self.sigma_max = config.Model.sigma_max.value
        self.sigmas = get_beta_schedule('exponential')

        # Define score functions for MRI and PET modalities
        self.mri_score_func = mri_score_func
        self.pet_score_func = pet_score_func

    def marginal_probability(self, x, t):
        """
        Computes the standard deviation and mean of the marginal probability distribution.
        The variance is computed exponentially using sigma_min and sigma_max.
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x # mean remains unchanged
        return mean, std
    
    def compute_diffusion(self, t):
        """
        Computes the diffusion coefficient for a given timestep t.
        Diffusion is determined using an exponential noise schedule.
        """
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        log_diff = tf.math.log(self.sigma_max) - tf.math.log(self.sigma_min)
        return sigma * tf.sqrt(tf.convert_to_tensor(2 * log_diff))
    
    def compute_mri_gradient_loss(self, x, t, mri):
        """
        Computes the gradient loss for MRI modality.
        It calculates the difference between the estimated score function and MRI data,
        scaled by the standard deviation of the marginal probability distribution.
        """
        with tf.GradientTape() as tape:
            x = tf.identity(x)  # ensures x is not modified directly
            tape.watch(x)  # enable gradients tracking for x

            std = self.marginal_probability(x, t)[1]  # calc stdev from marginal probability
            mri_grad_loss = (self.mri_score_func(x, t) - mri) ** 2 / (2 * std ** 2)

        grad = tape.gradient(mri_grad_loss, x) # compute gradient wrt x
        return grad
    
    def prior_sampling(self, shape):
        """
        Samples from the prior distribution, which is an isotropic Gaussian.
        Return x_T (pure noise) which is starting step of the reverse SDE.
        """
        return tf.random.normal(shape) * self.sigma_max
    
    def prior_logp(self, z):
        """
        Computes the log probability of a sample under the prior distribution.
        The prior is assumed to be a standard Gaussian with variance sigma_max.
        """
        shape = z.shape
        N = np.prod(shape[1:]) # computes dimensionality of each sample
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - tf.reduce_sum(z ** 2, axis=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def fwd_sde(self, x, t):
        """
        Forward-time SDE
            dx = f(x, t)dt + g(t)dw
        Where:
            f(x, t) = 0 (drift term)
            g(t) = sigma_min * (sigma_max / sigma_min)^t * sqrt(2 log(sigma_max / sigma_min))
        """
        drift = tf.zeros_like(x)
        diffusion = self.compute_diffusion(t)
        return drift, diffusion
    
    def fwd_discrete_2(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + g_i z_i
        Where:
            f_i(x_i) = 0 (drift term)
            g_i = sqrt(sigma_{i+1}^2 - sigma_i^2) is the diffusion coefficient
        """
        timestep = tf.cast(t * (self.N - 1) / self.T, tf.int64)
        sigma = tf.gather(self.sigmas, timestep)
        next_sigma = tf.gather(self.sigmas, tf.minimum(timestep + 1, self.N - 1))
        print("timestep: ", timestep, "N: ", self.N, "T: ", self.T, "sigma_t: ", sigma, "sigma_t+1: ", next_sigma)

        f = tf.zeros_like(x)
        g = tf.sqrt(next_sigma ** 2 - sigma ** 2)
        z = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        x = x + f + g * z
        return x
    
    def fwd_discrete(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + g_i z_i
        Where:
            f_i(x_i) = 0 (drift term)
            g_i = sigma_min * (sigma_max / sigma_min)^t is the diffusion coefficient
        """
        _, g = self.marginal_probability(x, t)
        z = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        x = x + g * z
        return x
    
    def reverse_sde(self, x, t, mri):
        """
        Reverse-time SDE
            dx = [f(x, t) - g(t)^2 delta_x log p_t(x, y)] dt + g(t) dW
        Where:
            f(x, t) = 0 (drift term) (because of variance exploding SDE)
            delta_x log p_t(x, y) is the score function
        """
        pet_grad = self.pet_score_func(x, t) # compute PET score func gradient
        mri_grad = self.compute_mri_gradient_loss(x, t, mri) # compute MRI gradient loss

        # update drift and diffusion
        diffusion = self.compute_diffusion(t)
        drift = -diffusion[:, None, None, None] ** 2 * (pet_grad + mri_grad) * (0.5 if self.deterministic_sampling else 1.0)
        diffusion = 0 if self.deterministic_sampling else diffusion

        # Euler-Maruyama step for reverse-time SDE
        dt = -1.0 / self.N
        z = tf.random.normal(shape=tf.shape(x), dtype=x.dtype) # sample noise
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * tf.sqrt(-dt) * z
        return x, x_mean

    def reverse_discrete(self, x, t, mri):
        """
        Discretizes the reverse SDE for stable numerical integration.
        x_i = x_{i+1} - g_{i+1}^2 s_theta(x_{i+1}, y, i+1) + g_{i+1} z_{i+1}
        Where:
            s_theta(x, y, t) is the score function
            g_i = sqrt(sigma_i^2 - sigma_{i-1}^2) is the diffusion coefficient
        """
        timestep = tf.cast(t * (self.N - 1) / self.T, tf.int64)
        sigma = tf.gather(self.sigmas, timestep)
        adjacent_sigma = tf.gather(self.sigmas, tf.maximum(timestep - 1, 0))

        g_i = tf.sqrt(sigma ** 2 - adjacent_sigma ** 2) # compute diffusion coefficient

        pet_grad = self.pet_score_func(x, t) # compute PET score function gradient
        mri_grad = self.mri_score_func(x, t) # compute MRI score function

        with tf.GradientTape() as tape:
            tape.watch(x)
            std = self.marginal_probability(x, t)[0]
            mri_loss = tf.reduce_mean((mri_grad - mri) ** 2 / 2 * std ** 2)
        mri_correction = tape.gradient(mri_loss, x)

        # compute reverse drift term using both PET and MRI gradients
        rev_f = -g_i[:, None, None, None] ** 2 * (pet_grad - mri_correction) * (0.5 if self.deterministic_sampling else 1.0)

        z = tf.random.normal(tf.shape(x), dtype=x.dtype)
        x_mean = x - rev_f
        x = x_mean + g_i[:, None, None, None] * z # add noise only if not in deterministic mode
        return x, x_mean