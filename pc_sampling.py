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
            - Compute the gradient (delta_x log p_t(x, y)) using PET & MRI scores
            - Compute step_size = (SNR * ||z|| / ||delta_x log p_t(x, y)||)^2
            - x_i = x_{i+1} + step_size * delta_x log p_t(x_i, y) + sqrt(2 * step_size) * z
        Returns:
            x: Updated sample after langevin correction
            x_mean: Mean of x before adding noise
        """
        for _ in range(self.n_steps):
            if config.Model.channel_merge:
                x_concat = tf.concat([x, mri], axis=1)
                grad = self.pet_score_func(x_concat, t) # delta_x log p_t(PET, MRI)
            else:
                pet_grad = self.sde.pet_score_func(x, t) # delta_x log p_t(PET)
                mri_grad = self.sde.mri_score_func(x, t) # delta_x log p_t(MRI)
                grad = pet_grad + mri_grad # combine both gradients

            # extract PET gradient only if channel merging is disabled
            if not config.Model.channel_merge:
                grad = grad[:, 0, :, :] # extract first channel
                grad = tf.expand_dims(grad, axis=1)

            # langevin correction step
            noise = tf.random.normal(tf.shape(x))
            grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=-1) # ||delta_x log p_t(x, y)||
            noise_norm = tf.norm(tf.reshape(noise, [tf.shape(noise)[0], -1]), axis=-1) # ||z||
            step_size = (self.snr * noise_norm / grad_norm) ** 2 # (SNR * ||z|| / ||grad||)^2

            x_mean = x + step_size[:, None, None, None] * grad  # x_mean = x + step_size * delta_x log p_t(x, y)
            x = x_mean + tf.sqrt(2 * step_size)[:, None, None, None] * noise  # x = x_mean + sqrt(2 * step_size) * z, adding remaining diffusion noise

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
        x = tf.concat([x, mri])
        # corrector step (Langevin dynamics)
        x, x_mean = corrector.update_func(x, vec_t, mri)
        # predictor step (Euler-Maruyama)
        x, x_mean = predictor.update_func(x, vec_t, mri) 

    return (x_mean if denoise else x)