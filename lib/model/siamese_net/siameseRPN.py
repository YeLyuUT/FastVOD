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
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class siameseRPN(nn.Module):
    def __init__(self, input_dim, anchor_scales, anchor_ratios, use_separable_correlation = False):
        super(siameseRPN, self).__init__()
        self.use_separable_correlation = use_separable_correlation
        self.din = input_dim  # get depth of input feature map, e.g., 512
        self.correlation_channel = 256
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

        # TODO this may be modified if used for other strides.
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # target branch.
        self.RPN_Conv_bbox = nn.Conv2d(self.din, self.correlation_channel, 3, 1, 1, bias=True)
        self.RPN_Conv_cls = nn.Conv2d(self.din, self.correlation_channel, 3, 1, 1, bias=True)

        # template branch.
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(self.din, self.correlation_channel*self.nc_score_out, 1, 1, 0)
        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(self.din, self.correlation_channel*self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0


        # conv_merge_1x1 is to merge separable convolution to achieve depth-wise separable convolution.
        if self.use_separable_correlation:
            self.conv_merge_1x1 = nn.Conv2d(self.din, 1, bias=False)
        else:
            self.bias = nn.Parameter(torch.zeros(self.din, 1, 1), requires_grad=True)

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

    def cross_correlation(self, target_feat, template_feat):
        '''
        Calculate cross correlation.
        :param target_feat: input feature map. It is of size (1,c,H,W)
        :param template_feat: convolution kernel. It is of size (N , K, C, kH, kW).
                    N is the number of templates. K is the number of anchors. C is the number of correlation channels.
        :return: correlation map.
        '''
        n_templates = template_feat.size(0) #N
        out_dim_w = template_feat.size(1) #K
        in_dim_w = template_feat.size(2) #C
        kh = template_feat.size(3)
        kw = template_feat.size(4)
        template_feat = template_feat.view(n_templates * out_dim_w, in_dim_w, kh, kw)
        H = target_feat.size(2)
        W = target_feat.size(3)

        out = nn.functional.conv2d(
            target_feat,
            template_feat,
            bias=None,
            stride=1,
            padding=int((template_feat.size(2) - 1) / 2),
            dilation=1,
            groups=1)

        out = out.view(n_templates, out_dim_w, H, W)
        out = out + self.bias

        return out

    def depth_wise_cross_correlation(self, target_feat, template_feat):
        '''
        Calculate depth-wise cross correlation.
        :param target_feat: input feature map. It is of size (1,c,H,W)
        :param template_feat: convolution kernel. It is of size (N * out_c, in_c, kH, kW).
                    N is the number of templates.
        :return: correlation map.
        '''
        n_templates = template_feat.size(0) #N
        out_dim_w = template_feat.size(1) #K
        in_dim_w = template_feat.size(2) #C
        kh = template_feat.size(3)
        kw = template_feat.size(4)
        template_feat = template_feat.view(n_templates * out_dim_w * in_dim_w, 1, kh, kw)
        H = target_feat.size(2)
        W = target_feat.size(3)

        out = nn.functional.conv2d(
            target_feat,
            template_feat,
            bias=None,
            stride=1,
            padding=int((template_feat.size(2)-1)/2),
            dilation=1,
            groups=in_dim_w)

        out = out.view(n_templates*out_dim_w, in_dim_w, H, W)
        out = self.conv_merge_1x1(out)
        out = out.view(n_templates, out_dim_w, H, W)
        out = out + self.bias
        return out

    def forward(self, input):
        '''
        The inputs are two tuples. One for each image.
        :param input holds data (target_feat, im_info, template_feat, gt_boxes, num_boxes)
                target_feat is of size (1, C, H, W)
                gt_boxes is a batch of gt_boxes for tracking, and is of size (N, 6). 6 represents: x1,y1,x2,y2,class,trackid.
                template_feat is of size (N, C, H, W).
        :return:
        '''
        if self.training:
            target_feat, im_info, template_feat, gt_boxes, num_boxes = input
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
        target_feat_cls = self.RPN_Conv_cls(target_feat)
        target_feat_bbox = self.RPN_Conv_bbox(target_feat)

        # template branch.
        template_feat_cls = self.RPN_cls_score(template_feat)
        template_feat_bbox = self.RPN_bbox_pred(template_feat)

        template_feat_cls = template_feat_cls.view(n_templates, self.nc_score_out, -1, template_feat_cls.size(2), template_feat_cls.size(3))
        template_feat_bbox = template_feat_bbox.view(n_templates, self.nc_bbox_out, -1, template_feat_cls.size(2), template_feat_cls.size(3))

        # correlation
        if self.use_separable_correlation:
            rpn_cls_score = self.depth_wise_cross_correlation(target_feat_cls, template_feat_cls)
            rpn_bbox_pred = self.depth_wise_cross_correlation(target_feat_bbox, template_feat_bbox)
        else:
            rpn_cls_score = self.cross_correlation(target_feat_cls, template_feat_cls)
            rpn_bbox_pred = self.cross_correlation(target_feat_bbox, template_feat_bbox)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
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
        return rois, self.rpn_loss_cls, self.rpn_loss_box

# TODO fix this.
def sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of template RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

    max_overlaps, gt_assignment = torch.max(overlaps, 2)

    batch_size = overlaps.size(0)
    num_proposal = overlaps.size(1)
    num_boxes_per_img = overlaps.size(2)

    offset = torch.arange(0, batch_size) * gt_boxes.size(1)
    offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

    labels = gt_boxes[:, :, 4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)

    labels_batch = labels.new(batch_size, rois_per_image).zero_()
    rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
    gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
    # Guard against the case when an image has fewer than max_fg_rois_per_image
    # foreground RoIs
    for i in range(batch_size):

        fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
        fg_num_rois = fg_inds.numel()

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
        bg_num_rois = bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
            # See https://github.com/pytorch/pytorch/issues/1868 for more details.
            # use numpy instead.
            # rand_num = torch.randperm(fg_num_rois).long().cuda()
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

            # Seems torch.rand has a bug, it will generate very large number and make an error.
            # We use numpy rand instead.
            # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            bg_inds = bg_inds[rand_num]

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = rois_per_image
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

            bg_inds = bg_inds[rand_num]
            bg_rois_per_this_image = rois_per_image
            fg_rois_per_this_image = 0
        else:
            raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

        # The indices that we're selecting (both fg and bg)
        keep_inds = torch.cat([fg_inds, bg_inds], 0)

        # Select sampled values from various arrays:
        labels_batch[i].copy_(labels[i][keep_inds])

        # Clamp labels for the background RoIs to 0
        if fg_rois_per_this_image < rois_per_image:
            labels_batch[i][fg_rois_per_this_image:] = 0

        rois_batch[i] = all_rois[i][keep_inds]
        rois_batch[i, :, 0] = i

        gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

    bbox_target_data = self._compute_targets_pytorch(
        rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])

    bbox_targets, bbox_inside_weights = \
        self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

    return labels_batch, rois_batch, bbox_targets, bbox_inside_weights



