import tensorflow as tf
from tqdm import tqdm

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

    def update_func(self, x, mri, t):
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
        for _ in tqdm(range(self.n_steps), desc="Correction Progress", total=self.n_steps):
            x_concat = tf.concat([x, mri], axis=-1)
            grad = self.sde.pet_score_func(x_concat, t) # delta_x log p_t(PET, MRI)

            # langevin correction step
            noise = tf.random.normal(tf.shape(x))
            grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=-1) # ||delta_x log p_t(x, y)||
            noise_norm = tf.norm(tf.reshape(noise, [tf.shape(noise)[0], -1]), axis=-1) # ||z||
            step_size = (self.snr * noise_norm / grad_norm) ** 2 # (SNR * ||z|| / ||grad||)^2

            x_mean = x + step_size[:, None, None, None] * grad  # x_mean = x + step_size * delta_x log p_t(x, y)
            x = x_mean + tf.sqrt(2 * step_size)[:, None, None, None] * noise  # x = x_mean + sqrt(2 * step_size) * z, adding remaining diffusion noise

        return x, x_mean

def sampler(sde, mri, snr, n_steps, eps=1e-3, denoise=True):
    """
    Predictor-Corrector (PC) sampler for diffusion model using VESDE.
    """
    predictor = EulerMaruyamaPredictor(sde)
    corrector = LangevinCorrector(sde, snr, n_steps)

    x = sde.prior_sampling(mri.shape)  # initialize from prior distribution
    x = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)  # add batch and channel dimension
    mri = tf.cast(mri, dtype=tf.float32)
    mri = tf.expand_dims(tf.expand_dims(mri, axis=-1), axis=0)  # add batch and channel dimension
    timesteps = tf.linspace(sde.T, eps, sde.N)  # time steps

    for i in tqdm(range(sde.N), desc="Sampling Progress", total=sde.N):
        t = timesteps[i]
        vec_t = tf.ones(x.shape[0]) * t
        # corrector step (Langevin dynamics)
        x, x_mean = corrector.update_func(x, mri, vec_t)
        # predictor step (Euler-Maruyama)
        x, x_mean = predictor.update_func(x, mri, vec_t) 

    return (x_mean if denoise else x)