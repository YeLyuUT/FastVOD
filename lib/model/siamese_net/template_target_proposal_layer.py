import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from model.siamese_net.template_proposal_layer import _TemplateProposalLayer
from model.siamese_net.weight_cropping_layer import weight_crop_layer

class _TemplateTargetProposalLayer(nn.Module):
    '''
    prepare template and target training pairs.
    '''

    def __init__(self):
        super(_TemplateTargetProposalLayer, self).__init__()
        self.template_proposal_layer = _TemplateProposalLayer()
        self.weights_extractor = weight_crop_layer() # hyper-parameters are defined by cfg.

    def forward(self, feats1, feats2, rpn_rois_1, gt_boxes_1, gt_boxes_2):
        '''

        :param feats1: size (N,C,H,W)
        :param feats2: size (N,C,H,W)
        :param rpn_rois_1: size (N,n,5) default n==256
        :param rpn_rois_2: size (N,n,5)
        :param gt_boxes_1: size (N,n,6) default n==128
        :param gt_boxes_2: size (N,n,6)
        :return:
        '''
        # feats 1 is template source, feats2 is target source.
        batch_size = feats1.size(0)
        # template_rois size (N,n,5) N:number of batches. n:number of rois.
        # template_labels size (N,n)
        # template_track_ids size (N,n)
        template_rois_all, template_labels_all, template_track_ids_all = self.template_proposal_layer(
            (rpn_rois_1,
             gt_boxes_1,
             feats1.size(3)/cfg.SIAMESE.WEIGHT_CROPPING_LAYER_SCALE,
             feats1.size(2)/cfg.SIAMESE.WEIGHT_CROPPING_LAYER_SCALE))
        template_weights_all = self.crop_weights_from_feats(feats1, template_rois_all).view(
            batch_size,
            template_rois_all.size(1),
            feats1.size(1),
            cfg.SIAMESE.TEMPLATE_SZ,
            cfg.SIAMESE.TEMPLATE_SZ)

        # for each item, it is (target_feat, template_weights, gt_boxes for each weight).
        # target gt_boxes should be of shape (n, 1, 6).
        rtv_training_tuples = []
        for idx in range(batch_size):
            nonzero_coords = torch.nonzero(template_labels_all[idx] > 0)
            fg_obj_inds = None
            if nonzero_coords.size(0) > 0:
                fg_obj_inds = nonzero_coords[:, 0]  # extracting rows.
            else:
                continue
            target_feat = feats2[idx:idx + 1]
            template_weights = template_weights_all[idx]
            template_track_ids = template_track_ids_all[idx]
            target_gt_boxes_all = gt_boxes_2[idx, :, :]
            target_gt_boxes = []

            for template_id in range(template_weights.size(0)):
                template_track_id = template_track_ids[template_id]
                has_gt = False
                if template_track_id >= 0:
                    for gt_box_2 in target_gt_boxes_all:
                        if gt_box_2[5] == template_track_id:
                            has_gt = True
                            target_gt_boxes.append(gt_box_2.view(1, -1))
                            break
                if not has_gt:
                    target_gt_boxes.append(gt_box_2.new_zeros(1, 6))

            target_gt_boxes = torch.stack(target_gt_boxes)

            template_weights = template_weights.index_select(0, fg_obj_inds)
            target_gt_boxes = target_gt_boxes.index_select(0, fg_obj_inds)
            rtv_training_tuples.append((target_feat, template_weights, target_gt_boxes))

        return rtv_training_tuples

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def crop_weights_from_feats(self, feats, rois):
        return self.weights_extractor(feats, rois.view(-1, 5))