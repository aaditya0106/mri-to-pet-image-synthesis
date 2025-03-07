from data.data import load_data
from model.ddpm import DDPM
from model.sde import VESDE
from loss import JDAMLoss
import config

import tensorflow as tf
import numpy as np
import time
import os

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

def train_eval_step(sde, model, optimizer, pet, mri, training=True):
    loss_klass = JDAMLoss(sde, train=training)
    if training:
        with tf.GradientTape() as tape:
            loss = loss_klass.compute_loss(model, pet, mri)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        loss = loss_klass.compute_loss(model, pet, mri)
    return loss


def train():
    # load data
    data = load_data()
    data = tf.data.Dataset.from_tensor_slices(data).batch(config.Training.batch_size.value)

    # initialize the model
    model = DDPM(activation=tf.keras.activations.swish)
    sde = VESDE(
    pet_score_func=lambda x, t: model(x, t, training=True),
    mri_score_func=lambda x, t: model(x, t, training=True),
    )

    # initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, clipnorm=1.0) # eps=1e-8, warmup_steps=5000

    # save checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # training loop
    for epoch in range(config.Training.epochs.value):
        total_loss = 0.
        for batch in data:
            start_time = time.time()
            mri = tf.expand_dims(tf.cast(batch[:, 0, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
            pet = tf.expand_dims(tf.cast(batch[:, 1, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
            loss = train_eval_step(sde, model, optimizer, pet, mri, training=True)
            total_loss += loss
            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{config.training.epochs}, Loss: {loss:.5f}, Mean Loss: {total_loss / (epoch + 1):.5f}, Time: {time.time() - start_time:.2f}s')

        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_manager.save()

if __name__ == '__main__':
    train()