import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

class siameseRPN(nn.Module):
    def __init__(self, input_dim):
        self.din = input_dim  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS

        # TODO this may be modified if used for other strides.
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.bias = nn.Parameter(torch.zeros(self.nc_bbox_out))

    def forward(self, input):
        '''
        The inputs are two tuples. One for each image.
        :param input holds data (conv_feat, im_info, gt_boxes, ws, num_gt_boxes)
                gt_boxes is a batch of gt_boxes for tracking, and is of size (batch_sz, 6). 6 represents: x1,y1,x2,y2, class, trackid.
                ws is a batch of weights extracted from the template images. (batch_sz, feat_dim, k, k)
                gt_boxes should have the same batch size as ws.
        :return:
        '''
        if self.training:
            input_feat, im_info, gt_boxes, ws = input
        else:
            input_feat, im_info, ws = input

    def depth_wise_cross_correlation(self, input_feat, w):
        '''
        Calculate depth-wise cross correlation.
        :param input_feat: input feature map.
        :param w: convolution kernel.
        :return: correlation map.
        '''
        assert input_feat.size(0)==1, 'the input feat should have batch size of 1.'
        assert w.size(0)==1, 'the weight should have output dim of 1.'
        assert input_feat.size(1)==w.size(1), 'the input dims of input feature and weight are not equal.'
        outs = []
        for i in range(w.size(1)):
            out = nn.functional.conv2d(
                input_feat[:, i:i + 1, :, :],
                w[:, i:i + 1, :, :],
                bias=None,
                stride=1,
                padding=int(w.size(2)-1/2),
                dilation=1,
                groups=1)
            outs.append(out)
        return torch.cat(outs, dim=1)


    def RPN(self, input_feat, ws, gt_boxes = None):
        # we use separable convolution to balance the two branches.
        # conv2d has the same number of groups as input_feat channels.
        for i in range(ws.size(0)):
            correlation_map = self.depth_wise_cross_correlation(input_feat, ws[i:i+1,:,:,:])


    def get_gt_pred_boxes(self):
        pass

    def get_filter(self, conv_feat, ):
        pass

    def get_loss(self):
        pass



