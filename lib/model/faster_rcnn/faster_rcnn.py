import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
import math
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _global_context_layer(nn.Module):
    def __init__(self, c_in, c_out, c_mid=256, ks=15):
        # the c_out should be set as multiple of pooled_size*pooled_size .
        super(_global_context_layer, self).__init__()

        self.c_out = c_out
        self.c_mid = c_mid
        self.ks = ks
        # define convolution ops.
        self.row_prev = nn.Conv2d(c_in, c_mid , kernel_size=(ks,1),padding=((ks-1)//2,0))
        self.row_post = nn.Conv2d(c_mid,c_out , kernel_size=(1,ks),padding=(0,(ks-1)//2))
        self.col_prev = nn.Conv2d(c_in, c_mid , kernel_size=(1,ks),padding=(0,(ks-1)//2))
        self.col_post = nn.Conv2d(c_mid,c_out , kernel_size=(ks,1),padding=((ks-1)//2,0))
        # two post conv-layers are specialized as the addition is applied after the layer.
        self._init_weights()
        self.row_post.weight.data = self.row_post.weight.data/2.0
        self.col_post.weight.data = self.col_post.weight.data/2.0

    def forward(self, feature):
        f_row = self.row_prev(feature)
        f_col = self.col_prev(feature)
        f_row = self.row_post(f_row)
        f_col = self.col_post(f_col)
        out = f_row+f_col
        #assert feature.size()[2:]==out.size()[2:], 'Check your global context layer.{}!={}'.format(feature.size()[2:],out.size()[2:])
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))

class _fasterRCNN(resnet):
    """ faster RCNN """
    def __init__(self, classes, num_layers=101, pretrained = False, class_agnostic = False, b_save_mid_convs = False):
        super(_fasterRCNN, self).__init__(classes, num_layers, pretrained, class_agnostic)
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.b_save_mid_convs = b_save_mid_convs
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.Conv4_feat = None
        self.rpn_rois = None

        if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
            print('RCNN uses Faster RCNN core.')
        elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
            print('RCNN uses RFCN Light Head core.')
            # The input channel is set mannually since we use resnet101 only.
            # c_out is set to 10*ps*ps. c_mid is set to 256.
            self.relu = nn.ReLU()
            core_depth = 10
            self.g_ctx = _global_context_layer(2048, core_depth * cfg.POOLING_SIZE * cfg.POOLING_SIZE, 256, 15)
            self.RCNN_psroi_pool = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0, cfg.POOLING_SIZE, core_depth)
            # fc layer for roi-wise prediction.
            # roi_mid_c in the original paper is 2048.
            roi_mid_c = 2048
            self.fc_roi = nn.Linear(core_depth * cfg.POOLING_SIZE * cfg.POOLING_SIZE, roi_mid_c)
        elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
            print('RCNN uses R-FCN core.')
            # define extra convolution layers for psroi input.
            tmp_c_in = 2048
            self.rfcn_cls = nn.Conv2d(tmp_c_in, self.n_classes * cfg.POOLING_SIZE * cfg.POOLING_SIZE, kernel_size=1)
            if self.class_agnostic:
                self.rfcn_bbox = nn.Conv2d(tmp_c_in, 4 * cfg.POOLING_SIZE * cfg.POOLING_SIZE, kernel_size=1)
            else:
                # Need to remove the background class for bbox regression.
                # Other circumstances are handled by torch.gather op later.
                self.rfcn_bbox = nn.Conv2d(tmp_c_in, 4 * (self.n_classes) * cfg.POOLING_SIZE * cfg.POOLING_SIZE, kernel_size=1)
            # define psroi layers
            self.RCNN_psroi_score = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0, cfg.POOLING_SIZE, self.n_classes)
            if self.class_agnostic:
                self.RCNN_psroi_bbox = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0, cfg.POOLING_SIZE, 4)
            else:
                self.RCNN_psroi_bbox = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0, cfg.POOLING_SIZE, 4*(self.n_classes))
            # define ave_roi_pooling layers.
            self.ave_pooling_bbox = nn.AvgPool2d(cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)
            self.ave_pooling_cls = nn.AvgPool2d(cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
            normal_init(self.fc_roi, 0, 0.01, cfg.TRAIN.TRUNCATED)

        elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
            normal_init(self.rfcn_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.rfcn_bbox, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        self.Conv4_feat = None
        self.rpn_rois = None

        batch_size = im_data.size(0)

        # reduce gt_boxe from length 6 to 5 if necessary.
        if gt_boxes is not None:
            if gt_boxes.size(2)==6:
                gt_boxes = gt_boxes[:,:,:5]
            gt_boxes = gt_boxes.data
            num_boxes = num_boxes.data

        im_info = im_info.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #c1 = self.Conv_block_1(im_data)
        #c2 = self.Conv_block_2(im_data)
        #c3 = self.Conv_block_3(im_data)
        #c4 = self.Conv_block_4(im_data)
        #base_feat = c4

        # feed base feature map tp RPN to obtain rois
        self.Conv4_feat = base_feat
        rois_rpn, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        self.rpn_rois = rois_rpn
        # if it is training phrase, then use ground truth bboxes for refinement.
        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.prepare_rois_for_training(rois_rpn, gt_boxes, num_boxes)
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois_rpn

        rois = Variable(rois)

        # The original implementation puts c5 to R-CNN.
        if not cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
            base_feat = self.RCNN_top(base_feat)
        '''
        if not cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
            c5 = self.Conv_block_5(c4)
            base_feat = c5
        else:
            c5 = None

        # register the tensors.
        if self.b_save_mid_convs is True:
            self.c3 = c3
            self.c4 = c4
            self.c5 = c5
        '''

        # convert base feat to roi predictions.
        bbox_pred, cls_prob, cls_score = self.base_feat_to_roi_pred(base_feat, rois, rois_label)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            # handle online hard example mining (OHEM) here.
            if cfg.TRAIN.OHEM is True:
                RCNN_loss_cls_tmp = F.cross_entropy(cls_score, rois_label, reduce=False)
                RCNN_loss_bbox_tmp = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, reduce=False)
                #assert RCNN_loss_cls_tmp.size()==RCNN_loss_bbox_tmp.size(), 'size not equal.{}!={}'.format(RCNN_loss_cls_tmp.size(),RCNN_loss_bbox_tmp.size())
                RCNN_loss_tmp = RCNN_loss_cls_tmp + RCNN_loss_bbox_tmp
                sorted_RCNN_loss_tmp, index = torch.sort(RCNN_loss_tmp, descending=True)

                # TODO add nms here.
                '''
                ordered_boxes = rois.view(-1, 5)[index,1:5]
                loss_boxes = torch.cat((ordered_boxes, sorted_RCNN_loss_tmp), 1)
                keep = nms(loss_boxes, cfg.TRAIN.OHEM_NMS).long().view(-1)
                keep = keep[:cfg.TRAIN.OHEM_BATCH_SIZE*batch_size]
                index.detach_()
                # we only keep the first <cfg.TRAIN.OHEM_BATCH_SIZE*batch_size> indexes.
                index = index[keep] 
                '''

                index.detach_()
                # redo forward to train hard examples only.
                # select first cfg.TRAIN.OHEM_BATCH_SIZE rois for training.
                index = index[:cfg.TRAIN.OHEM_BATCH_SIZE*batch_size]
                rois_view       = rois.view(-1,5).index_select(0, index)
                rois_label      = rois_label.index_select(0, index)
                rois_target     = rois_target.index_select(0, index)
                rois_inside_ws  = rois_inside_ws.index_select(0, index)
                rois_outside_ws = rois_outside_ws.index_select(0, index)

                bbox_pred, cls_prob, cls_score = self.base_feat_to_roi_pred(base_feat, rois_view, rois_label)

                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce = False)
                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, reduce = False)
                #print('1:', RCNN_loss_cls[:5])
                #print('2:', RCNN_loss_bbox[:5])
                #print('3:', sorted_RCNN_loss_tmp[:5])
                RCNN_loss_cls = RCNN_loss_cls.mean()
                RCNN_loss_bbox = RCNN_loss_bbox.mean()
            else:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

        if self.training:
          cls_prob = None
          bbox_pred = None
        else:
          cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
          bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def create_architecture(self):
        # _init_modules should go before _init_weights as some layers are newly initialized.
        self._init_modules()
        self._init_weights()

    def base_feat_to_roi_pred(self, base_feat, rois, rois_label):
        # handle base_feat
        if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
            # ctx op layer here.
            base_feat = self.g_ctx(base_feat)
            base_feat = self.relu(base_feat)
        elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
            base_feat_score = self.rfcn_cls(base_feat)
            base_feat_bbox = self.rfcn_bbox(base_feat)
        else:
            pass
        # handle pooling.
        if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
            # For RFCN, two pooled_feat blobs are needed. One for class score, one for bbox.
            assert cfg.POOLING_MODE =='pspool', 'R-FCN has to use ps-pooling. Please check your config file for correct input.'
            pooled_feat_score = self.RCNN_psroi_score(base_feat_score, rois.view(-1, 5))
            pooled_feat_bbox = self.RCNN_psroi_bbox(base_feat_bbox, rois.view(-1, 5))
            cls_score = self.ave_pooling_cls(pooled_feat_score)
            bbox_pred = self.ave_pooling_bbox(pooled_feat_bbox)
            cls_score = cls_score.view((cls_score.shape[0], cls_score.shape[1]))
            bbox_pred = bbox_pred.view((bbox_pred.shape[0], bbox_pred.shape[1]))
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1,4))
                bbox_pred = bbox_pred_select.squeeze(1)
            cls_prob = F.softmax(cls_score, 1)
            return bbox_pred, cls_prob, cls_score
        else:
            if cfg.POOLING_MODE == 'crop':
                # pdb.set_trace()
                # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
                grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
                grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
                pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
            elif cfg.POOLING_MODE == 'align':
                pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE =='pspool':
                pooled_feat = self.RCNN_psroi_pool(base_feat, rois.view(-1, 5))
            bbox_pred, cls_prob, cls_score = self.roi_wise_pred(pooled_feat, rois_label)
            return bbox_pred, cls_prob, cls_score

    def roi_wise_pred(self,pooled_feat,rois_label=None):
        # feed pooled features to top model
        if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
            roi_feat = self._head_to_tail(pooled_feat)
        elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
            roi_feat = pooled_feat.view(pooled_feat.size(0), -1)
            roi_feat = self.fc_roi(roi_feat)
        else:
            raise ValueError('Unknown RCNN type.')
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(roi_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(roi_feat)
        cls_prob = F.softmax(cls_score, 1)
        return bbox_pred, cls_prob, cls_score

    def prepare_rois_for_training(self, rois, gt_boxes, num_boxes):
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        return rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws