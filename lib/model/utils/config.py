from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory
__C.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.UPPER_BOUND = 2.0
__C.TRAIN.LOWER_BOUND = 0.5
# Trim size for input images to create minibatch
__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
# __C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

# Whether to tune the batch normalization parameters during training
__C.TRAIN.BN_TRAIN = False

# Online hard example mining. OHEM.
__C.TRAIN.OHEM = False
# OHEM batch size defines the number of selected samples of all original rois.
# The number should be smaller than batch size(__C.TRAIN.BATCH_SIZE).
__C.TRAIN.OHEM_BATCH_SIZE = 128
# Threshold for dedup.
__C.TRAIN.OHEM_NMS = 0.7

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000
# RPN threshold
__C.TEST.RPN_SCORE_THRESH = 0.0
__C.TEST.SIAM_RPN_SCORE_THRESH = 0.0
__C.TEST.NMS_CROSS_CLASS = 0.0
#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

# Core defines whether to use faster-rcnn or rfcn.
# This ugly design is resulted by the ugly inherent structure in the original faster-rcnn code.
# RESNET.CORE_CHOICE has to be either 'rfcn_lighthead' or 'faster_rcnn'
__C.RESNET.CORE_CHOICE = edict()
__C.RESNET.CORE_CHOICE.RFCN_LIGHTHEAD = 'rfcn_light_head'
__C.RESNET.CORE_CHOICE.FASTER_RCNN = 'faster_rcnn'
__C.RESNET.CORE_CHOICE.RFCN = 'rfcn'
# Set the used value here.
__C.RESNET.CORE_CHOICE.USE = __C.RESNET.CORE_CHOICE.FASTER_RCNN
__C.RESNET.GLOBAL_CONTEXT_RANGE = 15 #default 15, that is 15x15
__C.RESNET.GLOBAL_CONTEXT_OUT_DEPTH = 10 #default 10

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the first of all 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

#########################
#      Siamese RPN      #
#########################
__C.SIAMESE = edict()

# Anchor scales for RPN
__C.SIAMESE.ANCHOR_SCALES = [4,8,16,32]
# Anchor ratios for RPN
__C.SIAMESE.ANCHOR_RATIOS = [0.5,1,2]

# Proposal parameters.
__C.TRAIN.SIAMESE_RPN_PRE_NMS_TOP_N =12000
__C.TRAIN.SIAMESE_RPN_POST_NMS_TOP_N =2000
__C.TRAIN.SIAMESE_RPN_NMS_THRESH =0.7
__C.TRAIN.SIAMESE_RPN_MIN_SIZE =8
# Max number of objects for tracking training.
__C.TRAIN.SIAMESE_MAX_TRACKING_OBJ = 128
# For debug purpose.
__C.TRAIN.SIAMESE_ONLY = False

__C.TEST.SIAMESE_RPN_PRE_NMS_TOP_N =6000
__C.TEST.SIAMESE_RPN_POST_NMS_TOP_N =300
__C.TEST.SIAMESE_RPN_NMS_THRESH =0.7
__C.TEST.SIAMESE_RPN_MIN_SIZE =8

# Whether use deformable conv for feat trans or not.
__C.SIAMESE.USE_DCN = True
# Template selection threshold.
__C.SIAMESE.TEMPLATE_SEL_FG_THRESH = 0.7
# We do not need negative examples. So TEMPLATE_SEL_BG_THRESH_LO==TEMPLATE_SEL_BG_THRESH_HI
__C.SIAMESE.TEMPLATE_SEL_BG_THRESH_LO = 0.1
__C.SIAMESE.TEMPLATE_SEL_BG_THRESH_HI = 0.1
__C.SIAMESE.TEMPLATE_SEL_BATCH_SIZE = 128
# TODO Generate template from gt boxes.
__C.SIAMESE.TEMPLATE_GEN_FROM_GT_ITERS = 1

# Threshold used to select class template for tracking.
__C.SIAMESE.TEMPLATE_SEL_CLS_THRESH = 0.8
# The weight kernel size of the template.
__C.SIAMESE.TEMPLATE_SZ = 3
# Channel number before correlation. 256 as default.
__C.SIAMESE.NUM_CHANNELS_FOR_CORRELATION = 256
# Other RPN parameters.
__C.SIAMESE.RPN_BATCH_SIZE = 256
__C.SIAMESE.RPN_NMS_THRESH = 0.7
#__C.SIAMESE.RPN_PRE_NMS_TOP_N = 100
#__C.SIAMESE.RPN_POST_NMS_TOP_N = 20
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.SIAMESE.RPN_POSITIVE_OVERLAP = 0.5
# Select training samples for siameseRPN loss. RPN_NEGATIVE_OVERLAP_LO <= IOU <= RPN_NEGATIVE_OVERLAP_HI: negative example
__C.SIAMESE.RPN_NEGATIVE_OVERLAP_HI = 0.3
__C.SIAMESE.RPN_NEGATIVE_OVERLAP_LO = 0.01
# This should be 1.0. As we only propose positive training samples.
__C.SIAMESE.FG_FRACTION = 1.0
# SIAMESE.CROP_TYPE can be one of ('roi_align','center_crop')
__C.SIAMESE.CROP_TYPE = 'center_crop'
# SIAMESE DETECTION INFLUENCE WEIGHT. 1.0 means the same as the original weight.
__C.SIAMESE.DET_WEIGHT = 1.0
# SCORE THRESHOLD FOR TRACKING.
__C.SIAMESE.THRESH_FOR_TRACKING = 0.8

# Penalty control.
# TODO this may need further tuning.
__C.SIAMESE.HANNING_WINDOW_WEIGHT = 1.0
# The hanning window size is of instance size*HANNING_WINDOW_SIZE_FACTOR.
__C.SIAMESE.HANNING_WINDOW_SIZE_FACTOR = 1.0
__C.SIAMESE.PANELTY_K = 0.4
# Whether to use distance to penalize rpn box selection.
__C.SIAMESE.USE_POS_PRIOR_FOR_SEL = True
# Detach the siam training features or not.
__C.SIAMESE.DETACH_CONV1234 = False
__C.SIAMESE.DETACH_FEAT_FOR_TRACK = False
__C.SIAMESE.NORMALIZE_CORRELATION = True
__C.SIAMESE.WEIGHT_CROPPING_LAYER_SCALE = 1.0/16.0
__C.SIAMESE.USE_SEPARABLE_CORRELATION = False

__C.SIAMESE.NO_RPN_TRAINING = False
__C.SIAMESE.NO_RCNN_TRAINING = False

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 100

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Feature stride for RPN
__C.FEAT_STRIDE = [16, ]

__C.CUDA = False

__C.CROP_RESIZE_WITH_MAX_POOL = True

import pdb
def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
