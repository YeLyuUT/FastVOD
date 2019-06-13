import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils.config import cfg
from model.roi_align.modules.roi_align import RoIAlign
from model.roi_crop.modules.roi_crop import _RoICrop
from model.utils.net_utils import _affine_grid_gen

class weight_crop_layer(nn.Module):
    def __init__(self, din = 1024, spatial_scale=cfg.SIAMESE.WEIGHT_CROPPING_LAYER_SCALE):
        super(weight_crop_layer, self).__init__()
        self.crop_kernel = None
        self.spatial_shrinkage_layer = None
        self.spatial_scale = spatial_scale
        if cfg.SIAMESE.CROP_TYPE=='roi_align':
            # TODO careful with the stride. If moved to other layer, it should be changed.
            self.crop_kernel = RoIAlign(cfg.SIAMESE.TEMPLATE_SZ, cfg.SIAMESE.TEMPLATE_SZ, spatial_scale)
            #self.spatial_shrinkage_layer = nn.Conv2d(din, din, kernel_size=cfg.SIAMESE.TEMPLATE_SZ, stride=cfg.SIAMESE.TEMPLATE_SZ,groups=din)
        elif cfg.SIAMESE.CROP_TYPE == 'center_crop':
            self.h_sz = (cfg.SIAMESE.TEMPLATE_SZ - 1) / 2.0
            self.stride = 1.0 / spatial_scale
            self.crop_kernel = RoIAlign(cfg.SIAMESE.TEMPLATE_SZ, cfg.SIAMESE.TEMPLATE_SZ, spatial_scale)
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
        if self.spatial_shrinkage_layer is not None:
            normal_init(self.spatial_shrinkage_layer, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def forward(self, feats, rois):
        '''
        get the weights from the feats.
        :param feats: (N,C,H,W)
        :param rois: (n, 5)
        :return:
        '''
        if cfg.SIAMESE.CROP_TYPE == 'roi_align':
            cropped_feat = self.crop_kernel(feats, rois)
            out = cropped_feat
            #out = self.spatial_shrinkage_layer(cropped_feat)
        elif cfg.SIAMESE.CROP_TYPE == 'center_crop':
            rois_cntr_x = (rois[:,1]+rois[:,3])/2.0
            rois_cntr_y = (rois[:,2]+rois[:,4])/2.0
            rois[:, 1] = rois_cntr_x - self.h_sz * self.stride
            rois[:, 3] = rois_cntr_x + self.h_sz * self.stride
            rois[:, 2] = rois_cntr_y - self.h_sz * self.stride
            rois[:, 4] = rois_cntr_y + self.h_sz * self.stride
            cropped_feat = self.crop_kernel(feats, rois)
            out = cropped_feat
        else:
            raise ValueError('Not implemented.')
        return out

