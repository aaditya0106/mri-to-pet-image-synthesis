import tensorflow as tf

class JDAMLoss:
    """
    Implements the Score Matching Loss for PET-MRI training using VESDE
    """
    def __init__(self, sde, eps=1e-2, train=True):
        self.sde = sde
        self.train = train # true for training loss and false for evaluation loss
        self.eps = eps # smallest time step to sample from
        self.beta = 1.0 # beta parameter for score matching loss
        self.num_t = 5 # number of time steps to sample from
        self.t_vals = tf.linspace(0.1, 0.3, self.num_t) # time values for training

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
        if t is None:
            t_idx = tf.random.uniform([batch_size], minval=0, maxval=self.num_t, dtype=tf.int32) # random time step t uniformly in [eps, T]
            t = tf.gather(self.t_vals, t_idx)
        z = tf.random.normal(tf.shape(pet_clean)) # sample gaussian noise (brownian?)
        _, noise = self.sde.marginal_probability(pet_clean, t)
        x_t = pet_clean + noise * z # perturb data with noise
        noisy_input = tf.concat([x_t, mri_clean], axis=-1) # concatenate MRI data
        
        score = score_func(noisy_input, labels=t, training=self.train) # labels can be noise because f=0 in VESDE
        score = tf.cast(score, dtype=noise.dtype)
        z = tf.cast(z, dtype=noise.dtype)
        lambda_t = 1.0 / (noise ** 2)
        #lambda_t = tf.clip_by_value(lambda_t, clip_value_min=0.1, clip_value_max=1e4)

        losses = tf.square(noise * score + z) * lambda_t
        # losses = tf.reduce_mean(tf.reshape(losses, [tf.shape(losses)[0], -1]), axis=-1) # compute mean loss
        loss = tf.reduce_mean(losses)
        #tf.print("Raw loss value:", loss)
        return loss