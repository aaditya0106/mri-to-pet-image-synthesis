import tensorflow as tf
import config

class JDAMLoss:
    """
    Implements the Score Matching Loss for PET-MRI training using VESDE
    """
    def __init__(self, sde, eps=1e-3, train=True):
        self.sde = sde
        self.train = train # true for training loss and false for evaluation loss
        self.eps = eps # smallest time step to sample from
        self.beta = 1.0 # beta parameter for score matching loss

    def compute_loss_2(self, score_func, pet_clean, mri_clean, t=None):
        batch_size = tf.shape(pet_clean)[0]
        t = t or tf.random.uniform([batch_size], minval=self.eps, maxval=self.sde.T) # random time step t uniformly in [eps, T]
        z = tf.random.normal(tf.shape(pet_clean)) # sample gaussian noise
        _, noise = self.sde.marginal_probability(pet_clean, t)
        noisy_input = pet_clean + noise * z # perturb data with noise
        noisy_input = tf.concat([noisy_input, mri_clean], axis=-1) # concatenate MRI data
        
        score = score_func(noisy_input, labels=noise, training=self.train) # labels can be std (noise) because f=0 in VESDE

        losses = tf.square(noise * score + z) # you can duplicate noise if not merging channels
        losses = tf.reduce_mean(tf.reshape(losses, [tf.shape(losses)[0], -1]), axis=-1) # compute mean loss
        loss = tf.reduce_mean(losses)
        return loss

    def compute_loss(self, score_func, pet_clean, mri_clean, t=None):
        """
        Computes the score matching loss for PET-MRI training.
        L = E_t lambda(t) E_x(0) E_x(t)|x(0) [ || sigma(t) s_theta(x(t), y, t) + (x(t) - x(0)) / sigma(t) ||^2 ]
        Where:
            - sigma(t) is noise scale at time t
            - lambda(t) = 1 / (sigma(t)^2) is weighting function
            - s_theta(x(t), y, t) is the score function which estimates delta_x log p_t(x, y)
            - x(0) is the original PET image
            - x(t) is noisy PET image at time t
        """
        batch_size = tf.shape(pet_clean)[0]
        t = t or tf.random.uniform([batch_size], minval=self.eps, maxval=self.sde.T) # random time step t uniformly in [eps, T]
        pet_noisy = self.sde.fwd_discrete(pet_clean, t)
        input_noisy = tf.concat([pet_noisy, mri_clean], axis=-1) # forward SDE step for PET-MRI

        sigma_t = self.sde.compute_diffusion(t)
        score = score_func(input_noisy, labels=sigma_t, training=self.train)
        lambda_t = 1 / (sigma_t ** 2)
        loss = lambda_t * tf.reduce_mean(tf.square(sigma_t * score + (pet_noisy - pet_clean) / sigma_t))
        return loss