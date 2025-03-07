import nibabel as nib
import os

def load_image(path):
    """
    Load image from path
    """
    img = nib.load(path).get_fdata()
    img = img[:, :, img.shape[2] // 2]
    return img

def load_data():
    """
    Load data
    """
    data = []
    path = '../t1_flair_asl_fdg_preprocessed/'
    files = os.listdir(path)
    for file in files:
        t1_path  = os.path.join(path, file, 'T1_MNI.nii.gz')
        t1_img   = load_image(t1_path)
        fdg_path = os.path.join(path, file, 'FDG_MNI.nii.gz')
        fdg_img   = load_image(fdg_path)
        data.append((t1_img, fdg_img))
    return data