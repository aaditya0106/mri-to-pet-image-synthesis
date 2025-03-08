from pc_sampling import sampler
from eval_metrics import PSNR
from train import *
import config

import tensorflow as tf
import numpy as np

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

def load_model():
    model, sde = get_models()
    model.load_weights('./checkpoints/ckpt-4')
    return model, sde

def generate(mri):
    model, sde = load_model()
    pet_pred = sampler(model, sde, mri, 1000)
    return pet_pred

def test():
    test_data = get_train_test_data()[1]
    model, sde = get_models()
    model.load_weights('./checkpoints/ckpt-4') # enter correct checkpoint to load best model

    best_psnr = 0.0
    predictions = []
    for i, batch in enumerate(test_data):
        mri = tf.expand_dims(tf.cast(batch[:, 0, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
        pet = tf.expand_dims(tf.cast(batch[:, 1, :, :], dtype=tf.float32), axis=-1) # adding dim for channel (b, h, w, c)
        pet_pred = sampler(model, sde, mri, 1000)
        psnr = PSNR(pet, pet_pred)
        best_psnr = max(best_psnr, psnr)
        print(f'image: {i+1}, test_psnr: {psnr:.5f}, best_psnr: {best_psnr:.5f}')
        predictions.append(pet_pred)

    return predictions