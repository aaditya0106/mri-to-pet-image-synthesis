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

    def compute_loss_2(self, score_func, pet_clean, mri_clean):
        batch_size = tf.shape(pet_clean)[0]
        t = tf.random.uniform([batch_size], minval=self.eps, maxval=self.sde.T) # random time step t uniformly in [eps, T]
        z = tf.random.normal(tf.shape(pet_clean)) # sample gaussian noise
        _, noise = self.sde.marginal_probability(pet_clean, t)
        noisy_input = pet_clean + noise * z # perturb data with noise
        noisy_input = tf.concat([noisy_input, mri_clean], axis=-1) # concatenate MRI data

        labels = self.sde.marginal_probability(tf.zeros_like(pet_clean), t)[1] 
        score = score_func(noisy_input, labels=labels, training=self.train) # compute score function # input self.train?

        losses = tf.square(score * noise[:, None, None, None] * z) # you can duplicate noise if not merging channels
        losses = tf.reduce_mean(tf.reshape(losses, [tf.shape(losses)[0], -1]), axis=-1) # compute mean loss
        loss = tf.reduce_mean(losses)
        return loss

    def compute_loss(self, score_func, pet_clean, mri_clean):
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
        t = tf.random.uniform([batch_size], minval=self.eps, maxval=self.sde.T) # random time step t uniformly in [eps, T]
        input_clean = tf.concat([pet_clean, mri_clean], axis=1)
        input_noisy = self.sde.fwd_discrete(input_clean, t) # forward SDE step for PET-MRI

        score = score_func(input_noisy, t)
        sigma_t = self.sde.compute_diffusion(t)
        lambda_t = 1 / (sigma_t ** 2)
        loss = lambda_t * tf.reduce_mean(tf.square(sigma_t * score + (input_noisy - input_clean) / sigma_t))
        return loss