from data.data import load_data
from model.ddpm import DDPM
from model.sde import VESDE
from loss import JDAMLoss
import config

import tensorflow as tf
import numpy as np
import time
import os
from tqdm import tqdm
import argparse

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

def get_train_test_data(split=0.9, path=config.Data.data_path.value):
    data = load_data(path)
    data = np.array(data)
    np.random.shuffle(data)
    split = int(len(data) * split)
    train_data = data[:split]
    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(config.Training.batch_size.value)
    test_data = data[split:]
    return train_data, test_data

def get_models():
    model = DDPM(activation=tf.keras.activations.swish)
    sde = VESDE(
        pet_score_func=lambda x, t: model(x, t, training=True),
        mri_score_func=lambda x, t: model(x, t, training=True),
    )
    return model, sde

def get_optimizer():
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, clipnorm=1.0) # eps=1e-8, warmup_steps=5000
    return optimizer

def get_chkpt_manager(optimizer, model, checkpoint_dir = config.Training.checkpoint_dir.value):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    return checkpoint_manager

def train_eval_step(sde, model, optimizer, pet, mri, training=True):
    loss_klass = JDAMLoss(sde, train=training)
    if training:
        with tf.GradientTape() as tape:
            loss = loss_klass.compute_loss_2(model, pet, mri)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        loss = loss_klass.compute_loss_2(model, pet, mri)
    return loss

def train(dataset_path=config.Data.data_path.value, checkpoint_dir=config.Training.checkpoint_dir.value):
    data, _             = get_train_test_data(path=dataset_path)                 # load data
    model, sde          = get_models()                          # initialize the model and sde
    optimizer           = get_optimizer()                       # initialize optimizer
    checkpoint_manager  = get_chkpt_manager(optimizer, model, checkpoint_dir=checkpoint_dir)   # initialize checkpoint manager

    # training loop
    for epoch in range(config.Training.epochs.value):
        total_loss = 0.
        cnt = 0
        with tqdm(total=len(data), desc=f'Epoch {epoch + 1}/{config.Training.epochs.value}', unit='batch') as pbar:
            for batch in data:
                start_time = time.time()
                mri = tf.expand_dims(tf.cast(batch[:, 0, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
                pet = tf.expand_dims(tf.cast(batch[:, 1, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
                loss = train_eval_step(sde, model, optimizer, pet, mri, training=True)
                total_loss += loss
                cnt += 1
                pbar.set_postfix(loss=loss.numpy())
                pbar.update(1)

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{config.Training.epochs.value}, Loss: {loss:.5f}, Mean Loss: {total_loss / cnt:.5f}, Time: {time.time() - start_time:.2f}s')

        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_manager.save()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MRI to PET image synthesis model.')
    parser.add_argument('--dataset_path', type=str, default=config.Data.data_path.value, help='Path to the dataset.')
    parser.add_argument('--checkpoint_dir', type=str, default=config.Training.checkpoint_dir.value, help='Directory to save checkpoints.')

    args = parser.parse_args()

    train(dataset_path=args.dataset_path, checkpoint_dir=args.checkpoint_dir)