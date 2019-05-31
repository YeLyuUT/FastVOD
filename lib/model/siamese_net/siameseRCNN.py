import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.siamese_net.template_target_proposal_layer import _TemplateTargetProposalLayer
from model.siamese_net.siameseRPN import siameseRPN
from model.utils.config import cfg

class _siameseRCNN(nn.Module):
    def __init__(self, classes, args):
        super(_siameseRCNN, self).__init__()
        self.RCNN = _fasterRCNN(classes, 101, pretrained=True, class_agnostic=args.class_agnostic, b_save_mid_convs=True)
        self.t_t_prop_layer = _TemplateTargetProposalLayer()
        self.siameseRPN_layer = siameseRPN( input_dim = 1024,
                                            anchor_scales = cfg.ANCHOR_SCALES,
                                            anchor_ratios = cfg.ANCHOR_RATIOS,
                                            use_separable_correlation = True)
        self.RCNN.create_architecture()

        # we only support cuda.
        self.siameseRPN_layer = self.siameseRPN_layer.cuda()
        self.RCNN = self.RCNN.cuda()

    def forward(self, input):
        if self.training:
            im_data_1, im_info_1, num_boxes_1, gt_boxes_1, im_data_2, im_info_2, num_boxes_2, gt_boxes_2 = input
            return self.forward_training(im_data_1, im_info_1, num_boxes_1, gt_boxes_1, im_data_2, im_info_2, num_boxes_2, gt_boxes_2)
        else:
            # testing.
            im_data, im_info, template_weights = input
            return self.forward_testing(template_weights, im_data, im_info, gt_boxes=None, num_boxes=None)

    def forward_testing(self, template_weights, im_data, im_info, gt_boxes, num_boxes):
        # Detection part.
        det_rois, det_cls_prob, det_bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        det_rois_label = self.RCNN(im_data, im_info, gt_boxes, num_boxes)

        if template_weights is not None:
            # Tracking part.
            #target_feat, template_weights, target_gt_boxes = None
            target_feat = self.RCNN.Conv4_feat
            target_gt_boxes = None
            input_v = (target_feat,
                       im_info,
                       template_weights)
            siam_rois, siam_scores, loss_cls, loss_box = self.siameseRPN_layer(input_v)
        else:
            siam_rois, siam_scores, loss_cls, loss_box = None, None, 0, 0
        return siam_rois, siam_scores, det_rois, det_rois_label, det_cls_prob, det_bbox_pred


    def forward_training(self,im_data_1, im_info_1, num_boxes_1, gt_boxes_1,
                     im_data_2, im_info_2, num_boxes_2, gt_boxes_2):
        ##################################
        #        Detection part          #
        ##################################
        # detection loss for image 1.
        rois_1, cls_prob_1, bbox_pred_1, \
        rpn_loss_cls_1, rpn_loss_box_1, \
        RCNN_loss_cls_1, RCNN_loss_bbox_1, \
        rois_label_1 = self.RCNN(im_data_1, im_info_1, gt_boxes_1, num_boxes_1)

        # c3_1, c4_1, c5_1 = RCNN.c_3, RCNN.c_4, RCNN.c_5
        conv4_feat_1 = self.RCNN.Conv4_feat
        rpn_rois_1 = self.RCNN.rpn_rois

        # detection loss for image 2.
        rois_2, cls_prob_2, bbox_pred_2, \
        rpn_loss_cls_2, rpn_loss_box_2, \
        RCNN_loss_cls_2, RCNN_loss_bbox_2, \
        rois_label_2 = self.RCNN(im_data_2, im_info_2, gt_boxes_2, num_boxes_2)

        # c3_2, c4_2, c5_2 = RCNN.c_3, RCNN.c_4, RCNN.c_5
        conv4_feat_2 = self.RCNN.Conv4_feat
        rpn_rois_2 = self.RCNN.rpn_rois

        ##################################
        #        Tracking part           #
        ##################################
        # define tracking loss here.
        tracking_losses_cls_ls = []
        tracking_losses_box_ls = []
        rtv_training_tuples = self.t_t_prop_layer(conv4_feat_1, conv4_feat_2, rpn_rois_1, gt_boxes_1, gt_boxes_2)
        for tpl_id in range(len(rtv_training_tuples)):
            target_feat, template_weights, target_gt_boxes = rtv_training_tuples[tpl_id]
            input_v = (target_feat,
                       im_info_2[tpl_id:tpl_id + 1],
                       template_weights,
                       target_gt_boxes,
                       1)
            rois, scores, rpn_loss_cls_siam, rpn_loss_box_siam = self.siameseRPN_layer(input_v)
            tracking_losses_cls_ls.append(rpn_loss_cls_siam)
            tracking_losses_box_ls.append(rpn_loss_box_siam)

        if len(tracking_losses_cls_ls) > 0:
            siamRPN_loss_cls = torch.mean(torch.stack(tracking_losses_cls_ls))
        else:
            siamRPN_loss_cls = rpn_loss_cls_2.new_zeros(1)
        if len(tracking_losses_box_ls) > 0:
            siamRPN_loss_box = torch.mean(torch.stack(tracking_losses_box_ls))
        else:
            siamRPN_loss_box = rpn_loss_box_2.new_zeros(1)

        rpn_loss_cls = (rpn_loss_cls_1.mean() + rpn_loss_cls_2.mean()) / 2
        rpn_loss_box = (rpn_loss_box_1.mean() + rpn_loss_box_2.mean()) / 2
        RCNN_loss_cls = (RCNN_loss_cls_1.mean() + RCNN_loss_cls_2.mean()) / 2
        RCNN_loss_bbox = (RCNN_loss_bbox_1.mean() + RCNN_loss_bbox_2.mean()) / 2
        rois_label = torch.cat((rois_label_1, rois_label_2),0)
        return rois, scores, rois_label, siamRPN_loss_cls, siamRPN_loss_box, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox




