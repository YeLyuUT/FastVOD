import torch
from model.utils.config import cfg
from torch.utils.data.dataloader import default_collate
import numpy as np

def collate_minibatch(list_of_inputs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    if isinstance(list_of_inputs[0], torch.Tensor):
        list_of_inputs = check_pad_tensor_data(list_of_inputs)
        out = None
        return torch.stack(list_of_inputs, 0, out=out)
    elif isinstance(list_of_inputs[0], list):
        transposed = zip(*list_of_inputs)
        return [collate_minibatch(b) for b in transposed]
    elif isinstance(list_of_inputs[0], tuple):
        transposed = zip(*list_of_inputs)
        return [collate_minibatch(b) for b in transposed]
    else:
        return default_collate(list_of_inputs)

def check_pad_tensor_data(list_of_tensors):
    tensor0 = list_of_tensors[0]
    if tensor0.dim() != 3:
        return list_of_tensors
    else:
        are_tensors_same_sz = True
        max_h = tensor0.size(1)
        max_w = tensor0.size(2)
        for i in range(1,len(list_of_tensors)):
            tensor = list_of_tensors[i]
            if are_tensors_same_sz is False or tensor0.size() != tensor.size():
                are_tensors_same_sz = False
                max_h = max(max_h, tensor.size(1))
                max_w = max(max_w, tensor.size(2))
        if are_tensors_same_sz is False:
            list_of_tensors = pad_image_data(list_of_tensors, torch.Size((tensor0.size(0),max_h,max_w)))
        return list_of_tensors

def pad_image_data(list_of_tensors, sz):
    '''
    :param list_of_tensors:
    :param sz: torch.Size of dim 3.
    :return:
    '''
    list_of_tensors = list(list_of_tensors)
    tensor0 = list_of_tensors[0]
    for i in range(len(list_of_tensors)):
        tnsr = list_of_tensors[i]
        sz_tnsr = tnsr.size()
        if sz_tnsr != sz:
            # padding the data if the sizes are not equal.
            new_tensor = tensor0.new_zeros(sz)
            new_tensor[:,:sz_tnsr[1],:sz_tnsr[2]] = tnsr
            list_of_tensors[i] = new_tensor
    list_of_tensors = tuple(list_of_tensors)
    return list_of_tensors
