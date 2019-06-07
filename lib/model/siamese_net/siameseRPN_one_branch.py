import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
#from model.rpn.proposal_layer import _ProposalLayer
from model.siamese_net.siam_proposal_layer import _SiamProposalLayer
from model.rpn.anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class siameseRPN_one_branch(nn.Module):
    def __init__(self, input_dim, anchor_scales, anchor_ratios, use_separable_correlation=False):
        super(siameseRPN_one_branch, self).__init__()
        self.use_separable_correlation = use_separable_correlation
        self.din = input_dim  # get depth of input feature map, e.g., 1024.
        self.correlation_channel = cfg.SIAMESE.NUM_CHANNELS_FOR_CORRELATION
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        # TODO add expand_factor to cfg.
        self.expand_factor = 64

        # TODO this may be modified if used for other strides.
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # target branch.
        self.RPN_Conv_target = nn.Conv2d(self.din, self.correlation_channel, 3, 1, 1, bias=True)
        self.target_BN = nn.BatchNorm2d(self.correlation_channel)
        # template branch.
        self.RPN_Conv_template = nn.Conv2d(self.din, self.correlation_channel*self.expand_factor, 3, 1, 1, bias=True)

        # prediction branch.
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(self.expand_factor, self.nc_score_out, 1, 1, 0)
        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(self.expand_factor, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _SiamProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # conv_merge_1x1 is to merge separable convolution to achieve depth-wise separable convolution.
        self.conv_merge_1x1_cls = None
        self.conv_merge_1x1_box = None
        if self.use_separable_correlation:
            self.conv_merge_1x1_cls = nn.Conv2d(self.correlation_channel, 1, 1, bias=False)
            self.conv_merge_1x1_box = nn.Conv2d(self.correlation_channel, 1, 1, bias=False)

        self.bias = nn.Parameter(torch.zeros(self.expand_factor, 1, 1), requires_grad=True)

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

        normal_init(self.RPN_Conv_target, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_Conv_template, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        if self.conv_merge_1x1_cls is not None:
            normal_init(self.conv_merge_1x1_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if self.conv_merge_1x1_box is not None:
            normal_init(self.conv_merge_1x1_box, 0, 0.01, cfg.TRAIN.TRUNCATED)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def cross_correlation(self, target_feat, template_feat, bias=None):
        '''
        Calculate cross correlation.
        :param target_feat: input feature map. It is of size (1,c,H,W)
        :param template_feat: convolution kernel. It is of size (N , K, C, kH, kW).
                    N is the number of templates. K is the number of anchors. C is the number of correlation channels.
        :return: correlation map.
        '''
        n_templates = template_feat.size(0)  # N
        out_dim_w = template_feat.size(1)  # K
        in_dim_w = template_feat.size(2)  # C
        kh = template_feat.size(3)
        kw = template_feat.size(4)
        template_feat = template_feat.view(n_templates * out_dim_w, in_dim_w, kh, kw)
        H = target_feat.size(2)
        W = target_feat.size(3)

        out = 0.1*nn.functional.conv2d(
            target_feat,
            template_feat,
            bias=None,
            stride=1,
            padding=int((template_feat.size(2) - 1) / 2),
            dilation=1,
            groups=1)

        out = out.view(n_templates, out_dim_w, H, W)
        if bias is not None:
            out = out + bias

        return out

    def depth_wise_cross_correlation_cls(self, target_feat, template_feat, bias=None):
        '''
        Calculate depth-wise cross correlation.
        :param target_feat: input feature map. It is of size (1,c,H,W)
        :param template_feat: convolution kernel. It is of size (N * out_c, in_c, kH, kW).
                    N is the number of templates.
        :return: correlation map.
        '''
        n_templates = template_feat.size(0)  # N
        out_dim_w = template_feat.size(1)  # K
        in_dim_w = template_feat.size(2)  # C
        kh = template_feat.size(3)
        kw = template_feat.size(4)
        template_feat = template_feat.view(n_templates * out_dim_w * in_dim_w, 1, kh, kw)
        H = target_feat.size(2)
        W = target_feat.size(3)

        out = 0.1*nn.functional.conv2d(
            target_feat,
            template_feat,
            bias=None,
            stride=1,
            padding=int((template_feat.size(2) - 1) / 2),
            dilation=1,
            groups=in_dim_w)

        out = out.view(n_templates * out_dim_w, in_dim_w, H, W)
        out = self.conv_merge_1x1_cls(out)
        out = out.view(n_templates, out_dim_w, H, W)
        if bias is not None:
            out = out + bias
        return out

    def depth_wise_cross_correlation_box(self, target_feat, template_feat, bias=None):
        '''
        Calculate depth-wise cross correlation.
        :param target_feat: input feature map. It is of size (1,c,H,W)
        :param template_feat: convolution kernel. It is of size (N * out_c, in_c, kH, kW).
                    N is the number of templates.
        :return: correlation map.
        '''
        n_templates = template_feat.size(0)  # N
        out_dim_w = template_feat.size(1)  # K
        in_dim_w = template_feat.size(2)  # C
        kh = template_feat.size(3)
        kw = template_feat.size(4)
        template_feat = template_feat.view(n_templates * out_dim_w * in_dim_w, 1, kh, kw)
        H = target_feat.size(2)
        W = target_feat.size(3)

        out = 0.1*nn.functional.conv2d(
            target_feat,
            template_feat,
            bias=None,
            stride=1,
            padding=int((template_feat.size(2) - 1) / 2),
            dilation=1,
            groups=in_dim_w)

        out = out.view(n_templates * out_dim_w, in_dim_w, H, W)
        out = self.conv_merge_1x1_box(out)
        out = out.view(n_templates, out_dim_w, H, W)
        if bias is not None:
            out = out + bias
        return out

    def forward(self, input):
        '''
        The inputs are two tuples. One for each image.
        :param input holds data (target_feat, im_info, template_feat, gt_boxes, num_boxes)
                target_feat is of size (1, C, H, W)
                gt_boxes is a batch of gt_boxes for tracking, and is of size (N, 1, 6). 6 represents: x1,y1,x2,y2,class,trackid.
                template_feat is of size (N, C, kH, kW).
        :return:
        '''
        if self.training:
            target_feat, im_info, template_feat, gt_boxes, num_boxes = input
            gt_boxes = gt_boxes[:, :, :5]
        else:
            target_feat, im_info, template_feat = input

        n_templates = template_feat.size(0)
        nC = template_feat.size(1)
        kh = template_feat.size(2)
        kw = template_feat.size(3)
        assert self.din == nC, 'The feature dims are not compatible.'
        assert nC == target_feat.size(1), 'The feature dims of template_feat and target_feat should be same.'
        assert target_feat.size(0) == 1, 'Input target_feat should have a batch size of 1.'

        # target branch.
        target_feat = self.RPN_Conv_target(target_feat)
        target_feat = self.target_BN(target_feat)
        # template branch.
        template_feat = self.RPN_Conv_template(template_feat)

        template_feat = template_feat.view(n_templates, self.expand_factor, -1, template_feat.size(2),
                                                   template_feat.size(3))

        # correlation
        if self.use_separable_correlation:
            rpn_feat = self.depth_wise_cross_correlation_cls(target_feat, template_feat, self.bias)
        else:
            rpn_feat = self.cross_correlation(target_feat, template_feat, self.bias)

        rpn_cls_score = self.RPN_cls_score(rpn_feat)
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_feat)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        im_info = im_info.expand((rpn_cls_prob.size(0), im_info.size(1)))
        rois, scores = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                  im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            batch_size = n_templates
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
        return rois, scores, self.rpn_loss_cls, self.rpn_loss_box





