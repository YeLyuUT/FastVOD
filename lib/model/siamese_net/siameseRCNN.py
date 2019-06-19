import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.siamese_net.template_target_proposal_layer import _TemplateTargetProposalLayer
#from model.siamese_net.siameseRPN import siameseRPN
from model.siamese_net.siameseRPN_one_branch import siameseRPN_one_branch as siameseRPN
from model.utils.config import cfg
from model.siamese_net.nms_tracking import trNMS
from torch.autograd import Variable
from model.dcn.modules.deform_conv import DeformConv
from random import shuffle

class block(nn.Module):
    def __init__(self, inplanes, planes):
        super(block, self).__init__()
        self.conv1 = self.conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.normal_init(self.conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)

        if cfg.SIAMESE.USE_DCN is True:
            self.conv_offsets_1 = self.conv3x3_offsets(inplanes)
            self.conv_offsets_2 = self.conv3x3_offsets(planes)
            self.normal_init(self.conv_offsets_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            self.normal_init(self.conv_offsets_2, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def conv3x3(self, inplanes, planes):
        if cfg.SIAMESE.USE_DCN is False:
            return nn.Conv2d(inplanes, planes, 3, 1, 1)
        else:
            return DeformConv(inplanes, planes, 3, 1, 1)

    def conv3x3_offsets(self, inplanes):
        return nn.Conv2d(inplanes, 3*3*2, 3, 1, 1, bias=False)

    def forward(self, x):
        residual = x
        if cfg.SIAMESE.USE_DCN is True:
            # deformable convolutions.
            offsets = self.conv_offsets_1(x)
            out = self.conv1(x, offsets)
            out = self.bn1(out)
            out = self.relu(out)

            offsets = self.conv_offsets_2(x)
            out = self.conv2(out, offsets)
            out = self.bn2(out)

            out += residual
            out = self.relu(out)
        else:
            # normal convolutions.
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += residual
            out = self.relu(out)
        return out

    def normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)

class _siameseRCNN(nn.Module):
    def __init__(self, classes, args):
        super(_siameseRCNN, self).__init__()
        self.RCNN = _fasterRCNN(classes, 101, pretrained=True, class_agnostic=args.class_agnostic, b_save_mid_convs=True)
        self.t_t_prop_layer = _TemplateTargetProposalLayer()
        self.siameseRPN_layer = siameseRPN( input_dim = 1024,
                                            anchor_scales = cfg.SIAMESE.ANCHOR_SCALES,
                                            anchor_ratios = cfg.SIAMESE.ANCHOR_RATIOS,
                                            use_separable_correlation = cfg.SIAMESE.USE_SEPARABLE_CORRELATION)
        self.RCNN.create_architecture()

        # Tracking feature branch.
        self.track_feat_trans = self._make_layer(block, 1024, 1024, 2).cuda()
        # we only support cuda.
        self.siameseRPN_layer = self.siameseRPN_layer.cuda()
        self.RCNN = self.RCNN.cuda()
        self.nms = trNMS()

    def _make_layer(self, block, inplanes, planes, blocks):
        layers = []
        layers.append(block(inplanes, planes))
        for i in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.training:
            im_data_1, im_info_1, num_boxes_1, gt_boxes_1, im_data_2, im_info_2, num_boxes_2, gt_boxes_2 = input
            return self.forward_training(im_data_1, im_info_1, num_boxes_1, gt_boxes_1, im_data_2, im_info_2, num_boxes_2, gt_boxes_2)
        else:
            # testing.
            im_data, im_info, template_weights, rois_tracking = input
            return self.forward_testing(rois_tracking, template_weights, im_data, im_info, gt_boxes=None, num_boxes=None)

    def forward_RCNN(self, base_feat, rois):
        bbox_pred, cls_prob, cls_score = self.RCNN.base_feat_to_roi_pred(base_feat, rois, None)
        return bbox_pred, cls_prob, cls_score

    def forward_testing(self, rois_tracking, template_weights, im_data, im_info, gt_boxes, num_boxes):
        '''

        :param rois_tracking: should be of size (N, 4+cls_num). eg. 35 for imagenetVID. (x1,y1,x2,y2,cls0,cls1,cls2,...,cls30).
        :param template_weights: should be a batch of template_weights (N, c_out,c_in,h,w).
        :param im_data:
        :param im_info:
        :param gt_boxes:
        :param num_boxes:
        :return:
        '''
        #################
        # Detection part.
        #################
        det_rois, det_cls_prob, det_bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        det_rois_label = self.RCNN(im_data, im_info, gt_boxes, num_boxes)

        #################
        # Tracking part.
        #################
        if rois_tracking is not None:
            # Tracking part.
            #target_feat, template_weights, target_gt_boxes = None
            target_feat = self.track_feat_trans(self.RCNN.Conv_feat_track)
            target_gt_boxes = None
            input_v = (target_feat,
                       im_info,
                       template_weights)
            siam_rois, siam_scores, loss_cls, loss_box = self.siameseRPN_layer(input_v)
            siam_rois = Variable(siam_rois)
            siam_scores = Variable(siam_scores)
            # NMS for siam rois.
            # TODO rois_tracking should be sliced.
            #print('siam_rois:',siam_rois)
            #print('siam_scores:',siam_scores)
            sel_siam_rois, sel_siam_scores = self.nms(rois=rois_tracking[:, :4], rpn_rois=siam_rois, scores=siam_scores)
            #print('sel_siam_rois:', sel_siam_rois)
            #print('sel_siam_scores:', sel_siam_scores)
            # Add batch indexes.
            batch_ids = sel_siam_rois.new_zeros((sel_siam_rois.size(0), 1))
            sel_siam_rois = torch.cat((batch_ids, sel_siam_rois), 1)
            tra_det_bbox_pred, tra_det_cls_prob, tra_det_cls_score = self.forward_RCNN(self.RCNN.base_feat_for_roi, sel_siam_rois)
            #print('tra_det_bbox_pred:',tra_det_bbox_pred)
            #print('tra_det_cls_prob:' ,tra_det_cls_prob)
            #print('tra_det_cls_score:',tra_det_cls_score)

            merged_probs = self.probs_for_tracking(
                fg_scores = sel_siam_scores,
                track_cls_probs = rois_tracking[:, 4:],
                tra_det_cls_probs = tra_det_cls_prob)
            siam_bbox_pred = tra_det_bbox_pred
            siam_cls_prob = merged_probs
            siam_rois = sel_siam_rois
        else:
            siam_rois, siam_bbox_pred, siam_cls_prob, loss_cls, loss_box = None, None, None, 0, 0

        return siam_rois, siam_bbox_pred, siam_cls_prob, det_rois, det_rois_label, det_cls_prob, det_bbox_pred

    def probs_for_tracking(self, fg_scores, track_cls_probs, tra_det_cls_probs):
        '''

        :param fg_scores: size (N, 1)
        :param track_cls_probs: size (N, 31)
        :param det_cls_probs: size (N, 31)
        :return:
        '''
        #print('fg_scores:',fg_scores.shape)
        #print('track_cls_probs:',track_cls_probs.shape)
        #print('tra_det_cls_probs:',tra_det_cls_probs.shape)
        #print('fg_scores:',fg_scores)
        mult_scores = fg_scores.repeat(1, track_cls_probs.size(1))
        #print('mult_scores:', mult_scores)
        sum_prob = track_cls_probs[:,1:].sum(dim=1, keepdim=True)+1e-5
        #print('mult_scores:', mult_scores.shape)
        #print('sum_prob:', sum_prob.shape)
        mult_scores[:, 1:] = mult_scores[:, 1:]*track_cls_probs[:,1:]/sum_prob
        mult_scores[:, 0] = 1.0-mult_scores[:, 0]

        merged_probs = mult_scores+tra_det_cls_probs*cfg.SIAMESE.DET_WEIGHT
        sum_merged_probs = merged_probs.sum(dim=1, keepdim=True)+1e-5
        merged_probs = merged_probs/sum_merged_probs
        return merged_probs

    def forward_training(self,im_data_1, im_info_1, num_boxes_1, gt_boxes_1,
                     im_data_2, im_info_2, num_boxes_2, gt_boxes_2):
        ##################################
        #        Detection part          #
        ##################################
        # detection loss for image 1.
        rois_1, cls_prob_1, bbox_pred_1, \
        rpn_loss_cls_1, rpn_loss_box_1, \
        RCNN_loss_cls_1, RCNN_loss_bbox_1, \
        rois_label_1 = self.RCNN(im_data_1, im_info_1, gt_boxes_1[:,:,:5], num_boxes_1)

        # c3_1, c4_1, c5_1 = RCNN.c_3, RCNN.c_4, RCNN.c_5
        if cfg.TRAIN.SIAMESE_ONLY is True:
            self.RCNN.Conv_feat_track.detach_()
        conv4_feat_1 = self.track_feat_trans(self.RCNN.Conv_feat_track)
        rpn_rois_1 = self.RCNN.rpn_rois

        # detection loss for image 2.
        rois_2, cls_prob_2, bbox_pred_2, \
        rpn_loss_cls_2, rpn_loss_box_2, \
        RCNN_loss_cls_2, RCNN_loss_bbox_2, \
        rois_label_2 = self.RCNN(im_data_2, im_info_2, gt_boxes_2[:,:,:5], num_boxes_2)

        # c3_2, c4_2, c5_2 = RCNN.c_3, RCNN.c_4, RCNN.c_5
        if cfg.TRAIN.SIAMESE_ONLY is True:
            self.RCNN.Conv_feat_track.detach_()
        conv4_feat_2 = self.track_feat_trans(self.RCNN.Conv_feat_track)
        rpn_rois_2 = self.RCNN.rpn_rois

        ##################################
        #        Tracking part           #
        ##################################
        # define tracking loss here.
        tracking_losses_cls_ls = []
        tracking_losses_box_ls = []
        rtv_training_tuples = self.t_t_prop_layer(conv4_feat_1, conv4_feat_2, rpn_rois_1, gt_boxes_1, gt_boxes_2)
        rois = None
        scores = None
        siamRPN_loss_cls = 0
        siamRPN_loss_box = 0
        shuffle(rtv_training_tuples)
        for tpl_id in range(len(rtv_training_tuples)):
            target_feat, template_weights, target_gt_boxes = rtv_training_tuples[tpl_id]
            # For memory issue, we randomly sample objects for training if there are too many objects to track.
            if target_gt_boxes.size(0) > cfg.TRAIN.SIAMESE_MAX_TRACKING_OBJ:
                perm = torch.randperm(target_gt_boxes.size(0))
                idx = perm[:cfg.TRAIN.SIAMESE_MAX_TRACKING_OBJ]
                target_gt_boxes = target_gt_boxes[idx]
                template_weights = template_weights[idx]
            input_v = (target_feat,
                       im_info_2[tpl_id:tpl_id + 1],
                       template_weights,
                       target_gt_boxes,
                       1)
            rois, scores, rpn_loss_cls_siam, rpn_loss_box_siam = self.siameseRPN_layer(input_v)
            if rpn_loss_cls_siam is not None:
                tracking_losses_cls_ls.append(rpn_loss_cls_siam)
            if rpn_loss_box_siam is not None:
                tracking_losses_box_ls.append(rpn_loss_box_siam)

        if len(tracking_losses_cls_ls) > 0:
            siamRPN_loss_cls = torch.mean(torch.stack(tracking_losses_cls_ls))
        else:
            siamRPN_loss_cls = None
        if len(tracking_losses_box_ls) > 0:
            siamRPN_loss_box = torch.mean(torch.stack(tracking_losses_box_ls))
        else:
            siamRPN_loss_box = None

        rpn_loss_cls = (rpn_loss_cls_1.mean() + rpn_loss_cls_2.mean()) / 2
        rpn_loss_box = (rpn_loss_box_1.mean() + rpn_loss_box_2.mean()) / 2
        RCNN_loss_cls = (RCNN_loss_cls_1.mean() + RCNN_loss_cls_2.mean()) / 2
        RCNN_loss_bbox = (RCNN_loss_bbox_1.mean() + RCNN_loss_bbox_2.mean()) / 2
        rois_label = torch.cat((rois_label_1, rois_label_2), 0)

        if siamRPN_loss_cls is not None:
            siamRPN_loss_cls = siamRPN_loss_cls.unsqueeze(0)
        else:
            siamRPN_loss_cls = rpn_loss_cls.new_zeros(1)
        if siamRPN_loss_box is not None:
            siamRPN_loss_box = siamRPN_loss_box.unsqueeze(0)
        else:
            siamRPN_loss_box = rpn_loss_cls.new_zeros(1)
        rpn_loss_cls = rpn_loss_cls.unsqueeze(0)
        rpn_loss_box = rpn_loss_box.unsqueeze(0)
        RCNN_loss_cls = RCNN_loss_cls.unsqueeze(0)
        RCNN_loss_bbox = RCNN_loss_bbox.unsqueeze(0)
        if cfg.TRAIN.SIAMESE_ONLY is True:
            rpn_loss_cls = rpn_loss_cls.new_zeros(1)
            rpn_loss_box = rpn_loss_cls.new_zeros(1)
            RCNN_loss_cls = RCNN_loss_cls.new_zeros(1)
            RCNN_loss_bbox = RCNN_loss_bbox.new_zeros(1)

        return rois_label, siamRPN_loss_cls, siamRPN_loss_box, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox
