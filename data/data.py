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
    img    = nib.load(path).get_fdata()
    z_dim  = img.shape[2]
    center = z_dim // 2
    start  = center - (slices // 2) # Compute start and end indices for slicing
    end    = center + (slices // 2) + (slices % 2) # If slices is odd, include the extra slice on the right
    imgs   = img[:, :, start:end] # shape: (H, W, S)
    imgs   = np.moveaxis(imgs, -1, 0) # Move the slice axis (third axis) to the front: (S, H, W)
    return imgs

def save_normalizer(t1_mean, t1_std, fdg_mean, fdg_std, norm_path='checkpoint/normalizer/normalizer.pkl'):
    """
    Save normalization parameters to a file.
    """
    # Ensure the directory exists.
    directory = os.path.dirname(norm_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    normalizer = {
        't1_mean' : t1_mean,
        't1_std'  : t1_std,
        'fdg_mean': fdg_mean,
        'fdg_std' : fdg_std
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

def load_data(data_path='../t1_flair_asl_fdg_preprocessed', slices=1, norm_path=config.Training.checkpoint_dir.value+'normalizer/normalizer.pkl'):

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
        t1_imgs  = load_image(t1_path, slices=slices)  # shape: (S, H, W)
        fdg_imgs = load_image(fdg_path, slices=slices)
        data    += list(zip(t1_imgs, fdg_imgs))

    # Check if a normalizer already exists; if not, compute and save it.
    t1_all, fdg_all = map(np.array, zip(*data))
    if not os.path.exists(norm_path):
        t1_mean, t1_std   = t1_all.mean(), t1_all.std()
        fdg_mean, fdg_std = fdg_all.mean(), fdg_all.std()
        save_normalizer(t1_mean, t1_std, fdg_mean, fdg_std, norm_path)
    else:
        norm = load_normalizer(norm_path)
        t1_mean, t1_std   = norm['t1_mean'], norm['t1_std']
        fdg_mean, fdg_std = norm['fdg_mean'], norm['fdg_std']
    
    # Normalize each slice using the computed parameters.
    normalized_data = np.stack([(t1_all - t1_mean) / t1_std, (fdg_all - fdg_mean) / fdg_std], axis=-1)
    return normalized_data