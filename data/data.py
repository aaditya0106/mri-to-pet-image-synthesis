import nibabel as nib
import numpy as np
import os
import config
import pickle

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

def save_normalizer(t1_mean, t1_std, fdg_mean, fdg_std, norm_path='normalizer.pkl'):
    """
    Save normalization parameters to a file.
    """
    normalizer = {
        't1_mean': t1_mean,
        't1_std': t1_std,
        'fdg_mean': fdg_mean,
        'fdg_std': fdg_std
    }
    with open(norm_path, 'wb') as f:
        pickle.dump(normalizer, f)
    print(f"Normalizer saved to {norm_path}")

def load_normalizer(norm_path='normalizer.pkl'):
    """
    Load normalization parameters from a file.
    """
    with open(norm_path, 'rb') as f:
        normalizer = pickle.load(f)
    return normalizer

def load_data(data_path='t1_flair_asl_fdg_preprocessed', slices=1, norm_path=config.Training.checkpoint_dir+'/normalizer/normalizer.pkl'):
    """
    Load data from subfolders in data_path. For each subject, load the T1 and FDG images,
    extract a centered block of slices (slices argument), pair corresponding slices,
    and normalize each modality using global normalization parameters saved to norm_path.
    
    If norm_path does not exist, it computes the global parameters over the dataset, saves them,
    and then applies normalization.
    
    Returns:
        A list of tuples: (normalized_T1_slice, normalized_FDG_slice)
    """
    data = []
    files = os.listdir(data_path)
    for file in files:
        t1_path  = os.path.join(data_path, file, 'T1_MNI.nii.gz')
        fdg_path = os.path.join(data_path, file, 'FDG_MNI.nii.gz')
        t1_img = load_image(t1_path, slices=slices)  # shape: (H, W, S)
        fdg_img = load_image(fdg_path, slices=slices)  # shape: (H, W, S)
        
        # Move the slice axis (third axis) to the front: (S, H, W)
        t1_slices = np.moveaxis(t1_img, -1, 0)
        fdg_slices = np.moveaxis(fdg_img, -1, 0)
        
        # Pair corresponding slices
        subject_slices = list(zip(t1_slices, fdg_slices))
        data += subject_slices

    # Check if a normalizer already exists; if not, compute and save it.
    if not os.path.exists(norm_path):
        # Stack all slices to compute global statistics.
        t1_all = np.array([pair[0] for pair in data])
        fdg_all = np.array([pair[1] for pair in data])
        
        t1_mean = t1_all.mean()
        t1_std  = t1_all.std()
        fdg_mean = fdg_all.mean()
        fdg_std  = fdg_all.std()
        
        save_normalizer(t1_mean, t1_std, fdg_mean, fdg_std, norm_path)
    else:
        norm = load_normalizer(norm_path)
        t1_mean, t1_std = norm['t1_mean'], norm['t1_std']
        fdg_mean, fdg_std = norm['fdg_mean'], norm['fdg_std']
    
    # Normalize each slice using the computed parameters.
    normalized_data = [
        ((t1 - t1_mean) / t1_std, (fdg - fdg_mean) / fdg_std) for t1, fdg in data
    ]
    
    return normalized_data