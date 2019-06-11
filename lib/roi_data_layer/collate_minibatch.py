from model.utils.config import cfg
from torch.utils.data.dataloader import default_collate
import numpy as np

def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])
    return Batch


def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
    return max_shape

def pad_image_data(list_of_blobs):
    max_shape = get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list