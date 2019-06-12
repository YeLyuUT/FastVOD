# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
try:
   import cPickle as pickle
   print('import cPickle')
except:
   import pickle
   print('import python pickle')
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
#from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.siamese_net.siameseRCNN import _siameseRCNN
from model.siamese_net.weight_cropping_layer import weight_crop_layer

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='hyper parameters to load',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--ckpt', dest='ckpt',
                      help='checkpoint to load model',
                      default='', type=str)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

def bbox_delta_to_pred_boxes(im_info, boxes, bbox_pred):
    box_deltas = bbox_pred.data
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        if args.class_agnostic:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)
        else:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    return pred_boxes

weight_cropper = weight_crop_layer().cuda()
def siam_weights_preparation(rois_tracking, base_feat):
    if rois_tracking is None:
        return None, None
    else:
        rois_tracking = Variable(base_feat.new_tensor(rois_tracking))
        boxes = rois_tracking[:,:4]
        batch_inds = boxes.new_zeros((boxes.size(0),1))
        boxes = torch.cat((batch_inds, boxes),dim=1)
        template_weights = weight_cropper(base_feat, boxes)
        return template_weights, rois_tracking

def prepare_rois_tracking(im_info, all_boxes, all_boxes_scores, frame_id, class_num, thresh=cfg.SIAMESE.THRESH_FOR_TRACKING):
    # class_num is 31 for imagenetVID.
    sel_boxes = []
    for j in range(1, class_num):
        if len(all_boxes[j][frame_id]) == 0:
            continue
        scored_boxes = all_boxes[j][frame_id].copy()
        scores = all_boxes_scores[j][frame_id].copy()
        assert len(scored_boxes)==len(scores), 'length of scored_boxes and length of scores should be the equal.'
        # TODO comment out the following for loop to accelerate predictions.
        scored_boxes[:, :4] = scored_boxes[:, :4] * im_info[-1]
        for b_id in range(len(scored_boxes)):
            assert scored_boxes[b_id, 4] == scores[b_id, j], 'scores not matched, please check your code.'
        inds = np.where(scored_boxes[:, 4]>thresh)[0]
        if len(inds)>0:
            sel_cls_boxes = np.concatenate((scored_boxes[inds,:4], scores[inds,:]), axis=1)
            sel_boxes.append(sel_cls_boxes)
        else:
            continue
    if len(sel_boxes)>0:
        rois_tracking = np.concatenate(sel_boxes, axis=0)
    else:
        rois_tracking = None
    return rois_tracking


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == 'imagenetDETVID':
      args.imdb_name = 'imagenetDETVID_train'
      args.imdbval_name = 'imagenetDETVID_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenetVID":
      args.imdb_name = 'imagenetVID_train'
      args.imdbval_name = 'imagenetVID_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenetVID_1_vid":
      args.imdb_name = 'imagenetVID_1_vid_train'
      # TODO imdbval is set to train set now.
      args.imdbval_name = 'imagenetVID_1_vid_train'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

  if args.cfg_file is None:
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # configure input output file names.
  if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
      pass
  elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
      pass
  elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
      pass
  else:
      raise ValueError('error.')

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)

  #print('cfg.RESNET.CORE_CHOICE.USE:',cfg.RESNET.CORE_CHOICE.USE)
  load_name_predix = cfg.RESNET.CORE_CHOICE.USE + '_siam'
  if cfg.TRAIN.OHEM is True:
      load_name_predix = load_name_predix+'_OHEM'
  load_name = os.path.join(input_dir, load_name_predix+'_{}.pth'.format(args.ckpt))

  # initilize the network here.
  if args.net == 'res101':
    RCNN = _siameseRCNN(imdb.classes, args)
  else:
    print("network is not defined")
    pdb.set_trace()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  RCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    RCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  #save_name = 'light_head_rcnn_10'
  save_name = load_name_predix
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_boxes_scores = [[[] for _ in xrange(num_images)]
                       for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  RCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  template_weights = None
  rois_tracking = None
  vid_img_ind = 0

  incr_vid_img_ind = False
  for i in range(num_images):
      if incr_vid_img_ind is True:
          vid_img_ind += 1
          incr_vid_img_ind = False
      if i==imdb._structured_indexes[vid_img_ind][0]:
          print('Processing vid %d/%d.' % (vid_img_ind + 1, len(imdb._structured_indexes)))
          template_weights = None
          rois_tracking = None
      if i==imdb._structured_indexes[vid_img_ind][-1]:
          incr_vid_img_ind = True
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      input = im_data, im_info, template_weights, rois_tracking

      det_tic = time.time()
      siam_rois, siam_bbox_pred, siam_cls_prob, rois, rois_label, cls_prob, bbox_pred = RCNN(input)

      ###########################################
      # Get detection boxes.
      ###########################################
      if cfg.TEST.BBOX_REG:
          scores = cls_prob.data
          boxes = rois.data[:, :, 1:5]
          pred_boxes = bbox_delta_to_pred_boxes(im_info, boxes, bbox_pred)
          pred_boxes /= data[1][0][2].item()
          scores = scores.squeeze()
          pred_boxes = pred_boxes.squeeze()
          if siam_bbox_pred is not None:
              siam_scores = siam_cls_prob.data
              siam_boxes = siam_rois.data[:, 1:5]
              pred_siam_bbox = bbox_delta_to_pred_boxes(im_info, siam_boxes.unsqueeze(0), siam_bbox_pred.unsqueeze(0))
              pred_siam_bbox /= data[1][0][2].item()
              pred_siam_bbox = pred_siam_bbox.squeeze(0)
              # concatenate siambox and detbox.
              pred_boxes = torch.cat((pred_boxes, pred_siam_bbox), 0)
              scores = torch.cat((scores, siam_scores), 0)
      else:
          raise ValueError('Error. Should set cfg.TEST.BBOX_REG to True.')

      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      ###########################################
      # NMS for detection and save to all boxes.
      ###########################################
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            all_scores = scores[inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            all_scores = all_scores[order]
            ######### nms for each cls here ########
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            all_cls_scores = all_scores[keep.view(-1).long()]
            all_boxes[j][i] = cls_dets.cpu().numpy()
            all_boxes_scores[j][i] = all_cls_scores.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array
            all_boxes_scores[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]
                  all_boxes_scores[j][i] = all_boxes_scores[j][i][keep, :]
      ########
      # Get weights for the next iteration.
      ########
      # First, convert all_boxes to rois_tracking.#
      rois_tracking = prepare_rois_tracking(im_info[0], all_boxes, all_boxes_scores, frame_id=i,
                                            class_num=imdb.num_classes, thresh=cfg.SIAMESE.THRESH_FOR_TRACKING)
      base_feat = RCNN.track_feat_trans.cuda()(RCNN.RCNN.Conv_feat_track)
      template_weights, rois_tracking = siam_weights_preparation(rois_tracking, base_feat)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
