import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.roi_align.modules.roi_align import RoIAlignAvg

class weight_crop_layer(nn.Module):
    def __init__(self):
        self.crop_kernel = None
        if cfg.SIAMESE.CROP_TYPE=='roi_align':
            # TODO careful with the stride. If moved to other layer, it should be changed.
            self.crop_kernel = RoIAlignAvg(cfg.SIAMESE.TEMPLATE_SZ, cfg.SIAMESE.TEMPLATE_SZ, 1.0 / 16.0)
        else:
            raise ValueError('Not implemented.')

    def forward(self, feats, rois):
        '''
        get the weights from the feats.
        :param feats: (N,C,H,W)
        :param rois: (n, 5)
        :return:
        '''
        return self.crop_kernel(feats, rois)

