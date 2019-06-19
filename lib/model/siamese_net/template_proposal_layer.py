# propose template images.
import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class _TemplateProposalLayer(nn.Module):
    '''
    Propose template for siamese RPN.
    The functionality of this layer for training and testing are different.
    For training, it selects training samples from RPN output.
    The returned rois and track_ids are used for training sample preparation.
    For testing, it selects predicted boxes with high enough score.
    '''
    def __init__(self):
        super(_TemplateProposalLayer, self).__init__()

    def forward(self, inputs):
        if self.training:
            # max_w, max_h are for boxes clipping in randomize_boxes func.
            all_rois, gt_boxes, max_w, max_h = inputs
            rois, labels, track_ids = self.propose_template_training(all_rois, gt_boxes, max_w, max_h)
            return rois, labels, track_ids
        else:
            bbox_pred, cls_prob, track_ids = inputs
            bbox_pred, cls_prob, track_ids = self.propose_template_testing(bbox_pred, cls_prob, track_ids)
            return bbox_pred, cls_prob, track_ids

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def propose_template_testing(self, bbox_pred, cls_prob, track_ids):
        sel_ind = torch.where(cls_prob>cfg.SIAMESE.TEMPLATE_SEL_CLS_THRESH)[0]
        return bbox_pred[sel_ind], cls_prob[sel_ind], track_ids[sel_ind]

    def clip_vector_val(self, vals, min, max):
        sz = vals.size()
        vals = vals.view(-1)
        nzrs = torch.nonzero(vals < min)
        if len(nzrs>0):
            vals[nzrs[0]] = min
        nzrs = torch.nonzero(vals > max)
        if len(nzrs>0):
            vals[nzrs[0]] = max
        return vals.view(sz)

    def randomize_boxes(self, boxes, max_w, max_h, c_std=0.3, s_std=0.5):
        '''

        :param boxes: size of (N, n, 5)
        :param c_std: lower and upper bound of center position changing ratio.
        :param s_std: lower and upper bound of size changing ratio.
        :return:
        '''
        new_boxes = boxes.clone().detach()

        x = (new_boxes[:, :, 1] + new_boxes[:, :, 3]) / 2.0
        y = (new_boxes[:, :, 2] + new_boxes[:, :, 4]) / 2.0
        w = (new_boxes[:, :, 3] - new_boxes[:, :, 1])
        h = (new_boxes[:, :, 4] - new_boxes[:, :, 2])
        new_x = x.new_zeros(x.size()).uniform_(-c_std, c_std) * w + x
        new_y = y.new_zeros(y.size()).uniform_(-c_std, c_std) * h + y
        new_w = w.new_zeros(w.size()).uniform_(-s_std, s_std).exp() * w
        new_h = h.new_zeros(h.size()).uniform_(-s_std, s_std).exp() * h

        new_x1 = new_x - new_w / 2.0
        new_y1 = new_y - new_h / 2.0
        new_x2 = new_x + new_w / 2.0
        new_y2 = new_y + new_h / 2.0

        new_boxes[:, :, 1] = self.clip_vector_val(new_x1, 0., max_w-1.)
        new_boxes[:, :, 2] = self.clip_vector_val(new_y1, 0., max_h-1.)
        new_boxes[:, :, 3] = self.clip_vector_val(new_x2, 0., max_w-1.)
        new_boxes[:, :, 4] = self.clip_vector_val(new_y2, 0., max_h-1.)

        return new_boxes

    def propose_template_training(self, all_rois, gt_boxes, max_w, max_h):
        num_images = 1
        rois_per_image = int(cfg.SIAMESE.TEMPLATE_SEL_BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.SIAMESE.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        gt_boxes_append = gt_boxes.new_zeros((gt_boxes.size()[0], gt_boxes.size()[1], 5))
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        for ibatch in range(len(gt_boxes)):
            gt_boxes_append[ibatch, :, 0] = ibatch
        gen_temp_tpls = []
        # TODO here we do not use rpn rois, but directly sample from GTs.
        all_rois = None
        for i in range(cfg.SIAMESE.TEMPLATE_GEN_FROM_GT_ITERS):
            # Include ground-truth boxes in the set of candidate rois
            if all_rois is None:
                all_rois = self.randomize_boxes(gt_boxes_append, max_w, max_h)
            else:
                # Use rpn rois + GTs boxes.
                all_rois = torch.cat([all_rois], 1)

            rois, labels, track_ids = self.sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image)
            gen_temp_tpls.append((rois, labels, track_ids))
        ### Add GT here.###
        rois, labels, track_ids = self.sample_rois_pytorch(gt_boxes_append, gt_boxes, fg_rois_per_image, rois_per_image)
        gen_temp_tpls.append((rois, labels, track_ids))
        ###################
        rois = torch.cat([gen[0] for gen in gen_temp_tpls], 1)
        labels = torch.cat([gen[1] for gen in gen_temp_tpls], 1)
        track_ids = torch.cat([gen[2] for gen in gen_temp_tpls], 1)
        return rois, labels, track_ids

    def sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
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

        labels = gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        track_id = gt_boxes[:, :, 5].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        track_id_batch = track_id.new(batch_size, rois_per_image).zero_()-1

        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.SIAMESE.TEMPLATE_SEL_FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.SIAMESE.TEMPLATE_SEL_BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.SIAMESE.TEMPLATE_SEL_BG_THRESH_LO) |
                                    (max_overlaps[i] < 0)).view(-1)
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
                print('overlaps:', overlaps.shape)
                print('max_overlaps:', max_overlaps)
                raise ValueError("template proposal layer bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            track_id_batch[i].copy_(track_id[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
                track_id_batch[i][fg_rois_per_this_image:] = -1

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i

        return rois_batch, labels_batch, track_id_batch


