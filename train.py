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
    np.random.shuffle(data)
    split = int(len(data) * split)
    train_data = data[:split]
    train_data = train_data[ : (len(train_data) // config.Training.batch_size.value) * config.Training.batch_size.value ]
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, clipnorm=1.0) # eps=1e-8, warmup_steps=5000
    return optimizer

def get_chkpt_manager(optimizer, model, checkpoint_dir=config.Training.checkpoint_dir.value, secondary_checkpoint_dir=config.Training.secondary_checkpoint_dir.value):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(secondary_checkpoint_dir, exist_ok=True)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    secondary_checkpoint_manager = tf.train.CheckpointManager(checkpoint, secondary_checkpoint_dir, max_to_keep=5)
    
    return checkpoint_manager, secondary_checkpoint_manager

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
    checkpoint_manager, s_checkpoint_manager  = get_chkpt_manager(optimizer, model, checkpoint_dir=checkpoint_dir, secondary_checkpoint_dir=config.Training.secondary_checkpoint_dir.value)   # initialize checkpoint manager

    # training loop
    for epoch in range(config.Training.epochs.value):
        total_loss = 0.
        cnt = 0
        start_time = time.time()
        with tqdm(total=len(data), desc=f'Epoch {epoch + 1}/{config.Training.epochs.value}', unit='batch') as pbar:
            for batch in data:
                start_time = time.time()
                mri = tf.expand_dims(tf.cast(batch[:, :, :, 0], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
                pet = tf.expand_dims(tf.cast(batch[:, :, :, 1], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
                loss = train_eval_step(sde, model, optimizer, pet, mri, training=True)
                total_loss += loss
                cnt += 1
                pbar.set_postfix(loss=loss)
                pbar.update(1)

        mean_loss = total_loss / (1 if cnt==0 else cnt)
        print(f'Epoch {epoch + 1}/{config.Training.epochs.value}, Mean Loss: {mean_loss:.5f}, Time: {time.time() - start_time:.2f}s')

        ckpt_mgr.save()
        s_ckpt_mgr.save()
        model.save_weights(checkpoint_dir + f'/model_weights_epoch:{epoch}.weights.h5')
        # save checkpoint every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     checkpoint_manager.save()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MRI to PET image synthesis model.')
    parser.add_argument('--dataset_path', type=str, default=config.Data.data_path.value, help='Path to the dataset.')
    parser.add_argument('--checkpoint_dir', type=str, default=config.Training.checkpoint_dir.value, help='Directory to save checkpoints.')

    args = parser.parse_args()

    train(dataset_path=args.dataset_path, checkpoint_dir=args.checkpoint_dir)