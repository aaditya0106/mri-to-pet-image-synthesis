from skimage.metrics import structural_similarity as SSIM 
import tensorflow as tf
import cv2

def SSIM(pet_generated, pet_original):
    """
    Calculate Structural Similarity Index score
    """
    if isinstance(pet_generated, tf.Tensor):
        pet_generated = pet_generated.numpy()
    if isinstance(pet_original, tf.Tensor):
        pet_original = pet_original.numpy()
    if len(pet_generated.shape) == 4:
        pet_generated = pet_generated[0, :, :, 0]
    if len(pet_original.shape) == 4:
        pet_original = pet_original[0, :, :, 0]

    data_range = pet_generated.max() - pet_generated.min()
    (score, diff) = SSIM(pet_generated, pet_original, full=True, data_range=data_range)
    diff = (diff * 255).astype("uint8")
    return score

def PSNR(pet_generated, pet_original):
    """
    Calculate Peak Signal to Noise Ratio.
    It's ratio between the max possible power of an image and the power of corrupting noise that affects its quality.
    """
    psnr = tf.image.psnr(pet_original, pet_generated, max_val=1.0)
    return psnr.numpy()[0]