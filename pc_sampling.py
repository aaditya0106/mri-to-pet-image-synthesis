import tensorflow as tf
import config

class EulerMaruyamaPredictor:
    """
    Predictor using Euler-Maruyama method for discrete reverse SDE.
    """
    def __init__(self, sde):
        self.sde = sde

    def update_func(self, x, t, mri):
        """
        Performs one Euler-Maruyama step for the reverse SDE.
        """
        return self.sde.reverse_discrete(x, t, mri) # discretized reverse SDE step

class LangevinCorrector:
    """
    Corrector using Langevin dynamics for sampling refinement.
    """
    def __init__(self, sde, snr, n_steps):
        self.sde = sde
        self.snr = snr  # signal-to-noise ratio
        self.n_steps = n_steps  # number of correction steps

    def update_func(self, x, t, mri):
        """
        Applies Langevin correction to refine sampling.
        Langevin correction step:
            - Compute the gradient (s_theta) using score function
            - Compute step_size = (SNR * ||z|| / ||s_theta||)^2
            - x_j = x_{j-1} + step_size * s_theta(x_i, y) + sqrt(2 * step_size) * z
        Returns:
            x: Updated sample after langevin correction
            x_mean: Mean of x before adding noise
        """
        for _ in range(self.n_steps):
            x_concat = tf.concat([x, mri], axis=1)
            score = self.pet_score_func(x_concat, t) # s_theta(PET, MRI)
            # langevin correction step
            z = tf.random.normal(tf.shape(x))
            z_norm = tf.norm(tf.reshape(z, [tf.shape(z)[0], -1]), axis=-1) # ||z||
            score_norm = tf.norm(tf.reshape(score, [tf.shape(score)[0], -1]), axis=-1) # ||s_theta||
            step_size = 2 * (self.snr * z_norm / score_norm) ** 2 # 2 * (SNR * ||z|| / ||score||)^2

            x_mean = x + step_size * score  # x_mean = x + step_size * s_theta
            x = x_mean + tf.sqrt(2 * step_size) * z  # x = x_mean + sqrt(2 * step_size) * z, adding remaining diffusion noise

        return x, x_mean

def sampler(sde, mri, shape, snr, n_steps, eps=1e-3, denoise=True):
    """
    Predictor-Corrector (PC) sampler for diffusion model using VESDE.
    """
    predictor = EulerMaruyamaPredictor(sde)
    corrector = LangevinCorrector(sde, snr, n_steps)

    x = sde.prior_sampling(shape)  # initialize from prior distribution
    timesteps = tf.linspace(sde.T, eps, sde.N)  # time steps

    for i in range(sde.N):
        t = timesteps[i]
        vec_t = tf.ones(shape[0]) * t
        # x = tf.concat([x, mri])
        # predictor step (Euler-Maruyama)
        x, x_mean = predictor.update_func(x, vec_t, mri) 
        # corrector step (Langevin dynamics)
        x, x_mean = corrector.update_func(x, vec_t, mri)

    return (x_mean if denoise else x)