from skimage.metrics import structural_similarity as ssim  # type: ignore
import cv2 # type: ignore

def SSIM(pet_generated, pet_original):
    """
    Calculate Structural Similarity Index score
    """
    pet_generated = cv2.cvtColor(pet_generated, cv2.COLOR_BGR2GRAY)
    pet_original = cv2.cvtColor(pet_original, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(pet_generated, pet_original, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def PSNR(pet_generated, pet_original):
    """
    Calculate Peak Signal to Noise Ratio.
    It's ratio between the max possible power of an image and the power of corrupting noise that affects its quality.
    """
    psnr = cv2.PSNR(pet_original, pet_generated)
    return psnr



# SMLD, or Score Matching Langevin Dynamics, is a sampling method used in the context of diffusion models
# It's rooted in the idea of gradually transforming a simple noise distribution into a complex data distribution through a 
# series of diffusion steps, and then reversing this process to generate new data samples.

# SMLD specifically uses score matching to estimate the gradient of the data distribution's log density (also known as the 
# score function) and Langevin dynamics to sample from the learned distribution.

def SMLD(pet_generated, pet_original):
    """
    Calculate score matching with langevin dynamics
    """
    pass