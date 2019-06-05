import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.roi_align.modules.roi_align import RoIAlignAvg

class weight_crop_layer(nn.Module):
    def __init__(self, din = 1024, spatial_scale=1.0 / 16.0):
        super(weight_crop_layer, self).__init__()
        self.crop_kernel = None
        if cfg.SIAMESE.CROP_TYPE=='roi_align':
            # TODO careful with the stride. If moved to other layer, it should be changed.
            self.crop_kernel = RoIAlignAvg(cfg.SIAMESE.TEMPLATE_SZ, cfg.SIAMESE.TEMPLATE_SZ, spatial_scale)
            self.spatial_shrinkage_layer = nn.Conv2d(din, din, kernel_size=cfg.SIAMESE.TEMPLATE_SZ, stride=cfg.SIAMESE.TEMPLATE_SZ)
        else:
            raise ValueError('Not implemented.')
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.spatial_shrinkage_layer, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def forward(self, feats, rois):
        '''
        get the weights from the feats.
        :param feats: (N,C,H,W)
        :param rois: (n, 5)
        :return:
        '''
        cropped_feat = self.crop_kernel(feats, rois)
        out = self.spatial_shrinkage_layer(cropped_feat)
        return out

