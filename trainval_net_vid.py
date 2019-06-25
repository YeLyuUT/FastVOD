# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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
import numpy.random as npr
import argparse
import pprint
import pdb
import time
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler

from roi_data_layer.prefetcher import data_prefetcher
from roi_data_layer.roidb_VID import combined_roidb_VID
from roi_data_layer.roibatchLoader_VID import roibatchLoader_VID
from roi_data_layer.collate_minibatch import collate_minibatch
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

#from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.siamese_net.siameseRCNN import _siameseRCNN


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='imagenetVID', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=15, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch size per video',
                      default=1, type=int)

  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

  parser.add_argument('--ckpt', dest='ckpt',
                      help='checkpoint to load model',
                      default='', type=str)

  parser.add_argument('--ckpt_det', dest='det_ckpt',
                      help='checkpoint to load detection model',
                      default='', type=str)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  parser.add_argument('--cfg',dest = 'cfg_file',
                      help='hyper parameters to load',
                      default=None, type=str)

  parser.add_argument('--no_save',dest='no_save',
                      help='whether to save model after every epoch.',
                      action='store_true')

  parser.add_argument('--snapshot_suffix',dest='snapshot_suffix',
                      help='suffix for save model name',
                      default='', type=str)

  args = parser.parse_args()
  return args

class batchSampler(BatchSampler):
    def __init__(self, sampler, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.sampler) / self.batch_size

class vid_plus_sampler(Sampler):
    def __init__(self, lmdb):
        '''
        This sampler samples batches from 1 video every time.
        :param train_size: the iteration per epoch.
        :param lmdb: the input lmdb.
        :param batch_size: number of video pairs for training.
        :param vid_per_cat: sampled video number for each category. Default 50.
        :param sample_gap_upper_bound: sample_gap_upper_bound is the maximum index gap to sample two images.
        '''
        zero_index = lmdb._zero_index
        image_index = lmdb._image_index
        vid_index_num = lmdb._vid_num
        idx_zero_index = 0
        samples = []
        for idx in range(len(image_index)):
            if idx==zero_index[idx_zero_index] and idx_zero_index<len(zero_index)-1:
                idx_zero_index+=1
                continue
            if idx>=lmdb._vid_num:
                # double the det number for balanced det and vid samples.
                samples.append((idx, idx))
                samples.append((idx, idx))
            else:
                samples.append((idx-1, idx))
        random.shuffle(samples)
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

class sampler(Sampler):
    def __init__(self, train_size, lmdb, batch_size, vid_per_cat=50, sample_gap_upper_bound=10):
        '''
        This sampler samples batches from 1 video every time.
        :param train_size: the iteration per epoch.
        :param lmdb: the input lmdb.
        :param batch_size: number of video pairs for training.
        :param vid_per_cat: sampled video number for each category. Default 50.
        :param sample_gap_upper_bound: sample_gap_upper_bound is the maximum index gap to sample two images.
        '''
        assert train_size % batch_size == 0, 'train_size should be divided by batch_size.'
        self._index_gap_upper_bound = int(sample_gap_upper_bound / lmdb._gap)
        structured_indexes = lmdb._structured_indexes
        counter = 0
        samples = []
        while counter < train_size:
            # First, we sample the videos from each category.
            cat_idxs = list(range(30))
            sampled_vids_for_each_category = []
            for cat_idx in cat_idxs:
                vids = structured_indexes[cat_idx]
                if len(vids) > 0:
                    sampled_vids = random.sample(vids, vid_per_cat)
                    sampled_vids_for_each_category.append(sampled_vids)
                else:
                    sampled_vids_for_each_category.append([])
            # Next, we generate training sample indexes.
            for vid_id in range(vid_per_cat):
                cat_idxs = list(range(30))
                random.shuffle(cat_idxs)
                for cat_idx in cat_idxs:
                    vids = sampled_vids_for_each_category[cat_idx]
                    if len(vids) > 0:
                        vid = vids[vid_id]
                        for _ in range(batch_size):
                            # 0< _tmp_gap <self._index_gap_upper_bound+1
                            _tmp_gap = npr.randint(1, self._index_gap_upper_bound+1)
                            item = random.sample(vid[:_tmp_gap], 1)
                            item = (item[0], item[0] + _tmp_gap)
                            samples.append(item)
                            counter += 1
        self.samples = samples[:train_size]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


def create_tensor_holder():
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
    return im_data,im_info,num_boxes,gt_boxes

def get_CNN_params(model, lr):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    return params

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenet_10_imgs":
      args.imdb_name = "imagenet_10_imgs_train"
      args.imdbval_name = "imagenet_10_imgs_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenetVID_1_vid":
      args.imdb_name = 'imagenetVID_1_vid_train'
      args.imdbval_name = 'imagenetVID_1_vid_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenetVID":
      args.imdb_name = 'imagenetVID_train'
      args.imdbval_name = 'imagenetVID_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenetVID_PLUS":
      args.imdb_name = 'imagenetVID_PLUS_train'
      args.imdbval_name = 'imagenetVID_PLUS_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == 'imagenetDETVID':
      args.imdb_name = 'imagenetDETVID_train'
      args.imdbval_name = 'imagenetDETVID_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']


  if args.cfg_file is None:
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  #np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  # TODO change combined_roidb.
  imdb, roidb, ratio_list, ratio_index = combined_roidb_VID(args.imdb_name)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  #my_sampler = sampler(train_size = train_size, lmdb=imdb, batch_size=args.batch_size, vid_per_cat = 50, sample_gap_upper_bound = 100)
  my_sampler = vid_plus_sampler(lmdb=imdb)
  # TODO change back.
  # TODO change the dataloader and sampler.
  '''
  train_size = 21000
  vid_per_cat = 50
  if args.dataset == "imagenetVID_1_vid":
      train_size = 120
      vid_per_cat = 1
  my_sampler = sampler(
      train_size=train_size,
      lmdb=imdb,
      batch_size=args.batch_size,
      vid_per_cat=vid_per_cat,
      sample_gap_upper_bound=16)
  '''
  my_batch_sampler = batchSampler(sampler = my_sampler, batch_size=args.batch_size)

  dataset = roibatchLoader_VID(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=my_batch_sampler, num_workers=args.num_workers, collate_fn=collate_minibatch)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'res101':
      RCNN = _siameseRCNN(imdb.classes, args)
  else:
    print("network is not defined")
    pdb.set_trace()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  params = get_CNN_params(RCNN, lr)

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  assert args.cuda, 'Only cuda version is supported.'
  RCNN.cuda()

  assert not (args.ckpt is not '' and args.det_ckpt is not '')
  if args.ckpt is not '':
    load_name_predix = cfg.RESNET.CORE_CHOICE.USE+'_siam'
    # TODO add OHEM later.
    if cfg.TRAIN.OHEM is True:
      load_name_predix = load_name_predix + '_OHEM'
    load_name = os.path.join(output_dir, load_name_predix + '_{}.pth'.format(args.ckpt))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    # TODO
    args.start_epoch = checkpoint['epoch']
    RCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # TODO
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.det_ckpt is not '':
      load_name_predix = cfg.RESNET.CORE_CHOICE.USE
      load_name = os.path.join(output_dir, load_name_predix + '_{}.pth'.format(args.det_ckpt)).replace('VID_PLUS','DETVID')
      print("loading checkpoint %s" % (load_name))
      checkpoint = torch.load(load_name)
      #args.session = checkpoint['session']
      # TODO
      # args.start_epoch = checkpoint['epoch']
      RCNN.RCNN.load_state_dict(checkpoint['model'])
      # optimizer.load_state_dict(checkpoint['optimizer'])
      print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    RCNN = nn.DataParallel(RCNN)

  iters_per_epoch = int(len(my_batch_sampler))

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  data_iter = None
  im_data_1, im_info_1, num_boxes_1, gt_boxes_1 = create_tensor_holder()
  im_data_2, im_info_2, num_boxes_2, gt_boxes_2 = create_tensor_holder()
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    RCNN.train()
    loss_temp = 0
    start = time.time()

    if args.lr_decay_step>0 and epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    if data_iter is not None:
        del data_iter
    data_iter = iter(dataloader)

    for step in range(iters_per_epoch):
      data_1, data_2 = next(data_iter)

      im_data_1.data.resize_(data_1[0].size()).copy_(data_1[0])
      im_info_1.data.resize_(data_1[1].size()).copy_(data_1[1])
      gt_boxes_1.data.resize_(data_1[2].size()).copy_(data_1[2])
      num_boxes_1.data.resize_(data_1[3].size()).copy_(data_1[3])

      im_data_2.data.resize_(data_2[0].size()).copy_(data_2[0])
      im_info_2.data.resize_(data_2[1].size()).copy_(data_2[1])
      gt_boxes_2.data.resize_(data_2[2].size()).copy_(data_2[2])
      num_boxes_2.data.resize_(data_2[3].size()).copy_(data_2[3])

      RCNN.zero_grad()

      input = (im_data_1, im_info_1, num_boxes_1, gt_boxes_1, im_data_2, im_info_2, num_boxes_2, gt_boxes_2)
      rois_label, siamRPN_loss_cls, siamRPN_loss_box, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox = RCNN(input)
      loss = 0
      if not cfg.SIAMESE.NO_RPN_TRAINING:
          loss += rpn_loss_cls.mean() + rpn_loss_box.mean()
      if cfg.SIAMESE.NO_RCNN_TRAINING:
          loss += RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      if siamRPN_loss_cls is not None and siamRPN_loss_box is not None:
          loss += siamRPN_loss_cls.mean() + siamRPN_loss_box.mean()
      if loss==0:
          continue

      loss_temp += loss.item()
      # backward
      optimizer.zero_grad()
      loss.backward()
      # clip gradient
      clip_gradient(RCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
            if siamRPN_loss_cls is not None:
                loss_siam_cls = siamRPN_loss_cls.mean().item()
            if siamRPN_loss_box is not None:
                loss_siam_box = siamRPN_loss_box.mean().item()
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
        else:
            if siamRPN_loss_cls is not None:
                loss_siam_cls = siamRPN_loss_cls.item()
            else:
                loss_siam_cls = 0
            if siamRPN_loss_box is not None:
                loss_siam_box = siamRPN_loss_box.item()
            else:
                loss_siam_box = 0
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()

        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        print("\t\t\tsiam_rpn_cls: %.4f, siam_rpn_box: %.4f" \
              % (loss_siam_cls, loss_siam_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'loss_siam_cls':loss_siam_cls,
            'loss_siam_box':loss_siam_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    # prefix for the saved name.
    name_prefix = ''
    if cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.FASTER_RCNN:
        name_prefix = 'faster_rcnn'
    elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD:
        name_prefix = 'rfcn_light_head'
    elif cfg.RESNET.CORE_CHOICE.USE == cfg.RESNET.CORE_CHOICE.RFCN:
        name_prefix = 'rfcn'
    else:
        pass
    name_prefix = name_prefix+'_siam'
    if not args.no_save:
        save_name = os.path.join(output_dir, (name_prefix + '_{}_{}_{}'+args.snapshot_suffix+'.pth').format(args.session, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': RCNN.module.state_dict() if args.mGPUs else RCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()

