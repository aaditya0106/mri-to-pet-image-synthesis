import os
import time
import argparse

import numpy as np
import tensorflow as tf
from data.data import load_data
from model.unet import UNet
from model.sde import VESDE
from loss import JDAMLoss
import config
import matplotlib.pyplot as plt
from utils import plot_t_histogram

# Reproducibility
np.random.seed(config.seed)
tf.random.set_seed(config.seed)
#tf.keras.mixed_precision.set_global_policy('mixed_float16')


def get_overfit_data(path, batch_size):
    """
    Load a single MRI-PET pair and repeat it infinitely for overfitting/debugging.
    """
    data = load_data(path, slices=1)
    # single_pair = data  # take first pair
    # repeated = np.repeat(single_pair, batch_size, axis=0)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.batch(batch_size, drop_remainder=True)
    print (f"Data shape: {data.shape}")
    print (f"Data type: {data.dtype}")
    ds = ds.repeat()  # infinite stream
    return ds


def get_models_and_optimizer():
    model = UNet(activation=tf.nn.silu)
    sde = VESDE(score_func=model)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    #     tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9)
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9)
    return model, sde, optimizer


def get_checkpoint_manager(optimizer, model, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    return tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

@tf.function
def train_step(model, sde, optimizer, pet, mri, batch_size):
    fixed_t = tf.fill([batch_size], 0.5) # fixed time step for overfitting
    # tf.print("Fixed time step:", fixed_t)
    with tf.GradientTape() as tape:
        loss, t = JDAMLoss(sde, train=True).compute_loss(model, pet, mri)# , t=fixed_t)
    grads = tape.gradient(loss, model.trainable_variables)
    # gnorm = tf.linalg.global_norm(grads)
    # tf.print("grad norm:", gnorm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, t

def train(dataset_path=config.Data.data_path.value, checkpoint_dir=config.Training.checkpoint_dir.value, steps=5000, print_every=10):
    batch_size = config.Training.batch_size.value
    # Load a single MRI-PET pair and repeat it for overfitting
    data = load_data(dataset_path, slices=1, num_files=1)
    single_pair = data[0] if isinstance(data, (list, np.ndarray)) else data
    repeated = np.repeat(np.expand_dims(single_pair, 0), batch_size, axis=0)
    ds = tf.data.Dataset.from_tensor_slices(repeated)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    ds_iter = iter(ds)

    model, sde, optimizer = get_models_and_optimizer()
    ckpt_mgr = get_checkpoint_manager(optimizer, model, checkpoint_dir)

    running_loss = None
    losses = []
    for step in range(1, steps + 1):
        batch = next(ds_iter)
        # split channels: assume last dim has at least 2 channels [MRI, PET, ...]
        mri = tf.expand_dims(tf.cast(batch[:, :, :, 0], tf.float32), -1)
        pet = tf.expand_dims(tf.cast(batch[:, :, :, 1], tf.float32), -1)

        # compute and apply gradients
        loss, t = train_step(model, sde, optimizer, pet, mri, batch_size)
        loss_val = tf.cast(loss, tf.float32)
        losses.append(loss_val)

        if running_loss is None:
            running_loss = loss_val
            all_ts = t.numpy().flatten().tolist()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss_val
            all_ts.extend(t.numpy().flatten().tolist())

        if step % print_every == 0:
            t_np = np.array(all_ts)
            print(
                f"Step {step}/{steps}, Current Loss: {loss_val:.4f}, Running Avg: {running_loss:.4f}, "
                f"t range: [{t_np.min():.3f}, {t_np.max():.3f}], mean: {t_np.mean():.3f}, std: {t_np.std():.3f}"
            )
        
        if step % 100 == 0:
            # Save model weights every 100 steps
            ckpt_mgr.save(checkpoint_number=step)
            model.save_weights(os.path.join(checkpoint_dir, 'overfit_epoch_{step}.weights.h5'))
            print(f"Checkpoint saved at step {step}.")

    # Final save
    model.save_weights(os.path.join(checkpoint_dir, 'overfit_final_1.weights.h5'))
    print("Overfit training complete. Final weights saved.")
    # Plot loss
    plt.plot(losses)
    plt.title('Loss over training steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(checkpoint_dir, 'overfit_final_1_loss_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Overfit a single MRI-PET pair to debug model.'
    )
    parser.add_argument(
        '--dataset_path', type=str, default=config.Data.data_path.value,
        help='Path to your MRI-PET data file.'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default=config.Training.checkpoint_dir.value,
        help='Directory to save overfit checkpoints/weights.'
    )
    parser.add_argument(
        '--steps', type=int, default=2000,
        help='Total optimization steps to run.'
    )
    parser.add_argument(
        '--print_every', type=int, default=10,
        help='Print loss every N steps.'
    )
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        steps=args.steps,
        print_every=args.print_every
    )