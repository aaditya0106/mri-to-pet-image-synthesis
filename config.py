from enum import Enum

class Data(Enum):
    image_size = 64
    num_channels = 3
    num_bits = 8
    num_resolutions = 4
    attn_resolutions = [16]
    use_fp16 = False
    train_data = 'train'
    test_data = 'test'
    val_data = 'val'

class Model(Enum):
    nf = 64
    channel_mult = [1, 4, 8]
    num_resnet_blocks = 2
    attn_resolutions = [16]
    dropout = 0.0
    resamp_with_conv = True
    use_fp16 = False
    conditional = False

    # Noise schedule
    num_scales = 1000
    sigma_min = 0.0001
    sigma_max = 0.9999
    scale_by_sigma = False
    out_channels = 1

