from skimage.metrics import structural_similarity as SSIM 
import tensorflow as tf
import cv2

def SSIM(pet_generated, pet_original):
    """
    Calculate Structural Similarity Index score
    """
    pet_generated = cv2.cvtColor(pet_generated, cv2.COLOR_BGR2GRAY)
    pet_original  = cv2.cvtColor(pet_original, cv2.COLOR_BGR2GRAY)
    (score, diff) = SSIM(pet_generated, pet_original, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def PSNR(pet_generated, pet_original):
    """
    Calculate Peak Signal to Noise Ratio.
    It's ratio between the max possible power of an image and the power of corrupting noise that affects its quality.
    """
    psnr = tf.image.psnr(pet_original, pet_generated, max_val=1.0)
    return psnr