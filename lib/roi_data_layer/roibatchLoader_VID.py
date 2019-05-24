
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch_VID import get_minibatch_VID
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader_VID(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

  def _get_one_item(self, index):
      if self.training:
          # TODO simplify following.
          assert self.ratio_index[index]==index
          index_ratio = int(self.ratio_index[index])
      else:
          index_ratio = index

      # get the anchor index for current sample index
      # here we set the anchor index to the last one
      # sample in this group
      minibatch_db = [self._roidb[index_ratio]]
      blobs = get_minibatch_VID(minibatch_db, self._num_classes)
      # print('self._num_classes', self._num_classes)
      # print(type(blobs['data']))
      # print(type(blobs['im_info']))
      data = torch.from_numpy(blobs['data'])
      im_info = torch.from_numpy(blobs['im_info'])
      # we need to random shuffle the bounding box.
      data_height, data_width = data.size(1), data.size(2)
      if self.training:
          np.random.shuffle(blobs['gt_boxes'])
          gt_boxes = torch.from_numpy(blobs['gt_boxes'])

          ########################################################
          # padding the input image to fixed size for each group #
          ########################################################

          # NOTE1: need to cope with the case where a group cover both conditions. (done)
          # NOTE2: need to consider the situation for the tail samples. (no worry)
          # NOTE3: need to implement a parallel data loader. (no worry)
          # get the index range
          assert self._roidb[index_ratio]['need_crop']==0

          if len(gt_boxes>0):
              # check the bounding box:
              not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
              keep = torch.nonzero(not_keep == 0).view(-1)

              gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
              if keep.numel() != 0:
                  gt_boxes = gt_boxes[keep]
                  num_boxes = min(gt_boxes.size(0), self.max_num_box)
                  gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
              else:
                  num_boxes = 0
          else:
              # TODO here we hard code the dimension for gt_boxes to be 6. May change this dynamically.
              gt_boxes_padding = torch.FloatTensor(self.max_num_box, 6).zero_()
              num_boxes = 0

              # permute trim_data to adapt to downstream processing
          padding_data = data[0]
          padding_data = padding_data.permute(2, 0, 1).contiguous()
          im_info = im_info.view(3)

          return padding_data, im_info, gt_boxes_padding, num_boxes
      else:
          data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
          im_info = im_info.view(3)

          gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
          num_boxes = 0

          return data, im_info, gt_boxes, num_boxes

  def __getitem__(self, indexes):
    return [self._get_one_item(id) for id in indexes]

  def __len__(self):
    return len(self._roidb)
