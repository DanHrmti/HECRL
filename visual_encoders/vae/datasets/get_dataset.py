# datasets
from visual_encoders.vae.datasets.hecrl_ds import HECRLEnvImage, HECRLEnvVideo


def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "hecrl_env":
        dataset = HECRLEnvVideo(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    else:
        raise NotImplementedError
    return dataset


def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # load data
    if ds == "hecrl_env":
        dataset = HECRLEnvImage(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    else:
        raise NotImplementedError
    return dataset
