import nibabel as nib
import os

def load_image(path, i=0):
    """
    Load image from path
    """
    img = nib.load(path).get_fdata()
    img = img[:, :, (img.shape[2] // 2) + i]
    return img

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
        for i in range(-5, 6, 1):
            t1_img   = load_image(t1_path, 0)
            fdg_img   = load_image(fdg_path, 0)
            data.append((t1_img, fdg_img))
    return data