from enum import Enum
import platform
import os

seed = 42

def is_macos():
    return os.name == 'posix' and platform.system() == 'Darwin'

class Data(Enum):
    image_size                  = 128
    num_channels                = 2
    centered                    = False
    data_path                   = '../t1_flair_asl_fdg_preprocessed/' if is_macos() else '/content/drive/MyDrive/Project/MRItoPET/data/t1_flair_asl_fdg_preprocessed/'
    slices                      = 1
    
class Model(Enum):
    nf                          = 128
    channel_mult                = (1, 2, 2, 2)
    num_resnet_blocks           = 2
    attn_resolutions            = (16,)
    dropout                     = 0.0
    resamp_with_conv            = True
    conditional                 = True
    num_scales                  = 100
    sigma_min                   = 1e-3
    sigma_max                   = 0.999
    beta_min                    = 0.1
    beta_max                    = 20.
    scale_by_sigma              = False
    out_channels                = 1
    channel_merge               = True

class Training(Enum):
    batch_size                  = 1
    epochs                      = 100
    continuous                  = True
    reduce_mean                 = False
    joint                       = True
    checkpoint_dir              = './checkpoints/' if is_macos() else '/content/drive/MyDrive/Project/MRItoPET/checkpoints/'
    secondary_checkpoint_dir    = '../checkpoints/' if is_macos() else '/content/checkpoints/'
