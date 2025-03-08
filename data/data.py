import nibabel as nib
import numpy as np
import os
import config

def load_image(path, slices=1):
    """
    Load image from path and return a centered block of slices.
    
    If slices=1, returns a single (center) slice.
    If slices>1, returns that many consecutive slices centered in the volume.
    """
    img = nib.load(path).get_fdata()
    z_dim = img.shape[2]
    center = z_dim // 2
    # Compute start and end indices for slicing
    start = center - (slices // 2)
    # If slices is odd, include the extra slice on the right
    end = center + (slices // 2) + (slices % 2)
    return img[:, :, start:end]

def load_data(data_path = 't1_flair_asl_fdg_preprocessed'):
    """
    Load data
    """
    data = []
    path = data_path
    files = os.listdir(path)
    for file in files:
        t1_path  = os.path.join(path, file, 'T1_MNI.nii.gz')
        fdg_path = os.path.join(path, file, 'FDG_MNI.nii.gz')
        t1_img = load_image(t1_path, slices=config.Data.slices.value)  # shape: (H, W, S)
        fdg_img = load_image(fdg_path, slices=config.Data.slices.value)  # shape: (H, W, S)
        
        # Move the slice axis (third axis) to the front.
        # This yields arrays of shape (S, H, W)
        t1_slices = np.moveaxis(t1_img, -1, 0)
        fdg_slices = np.moveaxis(fdg_img, -1, 0)
        
        # Pair corresponding slices using zip without an explicit loop.
        # This returns a list of tuples where each tuple is (T1_slice, FDG_slice)
        subject_slices = list(zip(t1_slices, fdg_slices))
        
        # Extend the data list with the paired slices for this subject.
        data += subject_slices
    return data