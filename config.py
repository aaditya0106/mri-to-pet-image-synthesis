from enum import Enum

seed = 42

class Data(Enum):
    image_size                  = 128
    num_channels                = 2
    centered                    = False

class Model(Enum):
    nf                          = 128
    channel_mult                = (1, 2, 2, 2)
    num_resnet_blocks           = 2
    attn_resolutions            = (16,)
    dropout                     = 0.0
    resamp_with_conv            = True
    conditional                 = True
    num_scales                  = 1000
    sigma_min                   = 0.01
    sigma_max                   = 50.
    beta_min                    = 0.1
    beta_max                    = 20.
    scale_by_sigma              = False
    out_channels                = 1
    channel_merge               = True

class Training(Enum):
    batch_size                  = 1
    epochs                      = 1000
    likelihood_weighting        = False
    continuous                  = True
    reduce_mean                 = False
    joint                       = True