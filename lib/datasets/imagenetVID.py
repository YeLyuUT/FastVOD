from __future__ import print_function
# --------------------------------------------------------
# Copyright (c) 2019 University of Twente.
# Licensed under The MIT License [see LICENSE for details]
# Written by Ye Lyu.
# --------------------------------------------------------

import datasets
import datasets.imagenet
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
import random
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class imagenetVID(imdb):
    def __init__(self, image_set, devkit_path, data_path):
        imdb.__init__(self, 'imagenetVID_'+ image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = data_path
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))

        self._classes = ('__background__',)
        self._wnid = (0,)

        for i in xrange(30):
            self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)

        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        self._image_ext = ['.JPEG']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #print(index)
        #image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        image_path = index + self._image_ext[0]
        #print(image_path)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        # For VID dataset, we hard code the data number to sample.
        vid_num_per_cat = 100 # we use 100 videos for each category.
        num_img_per_vid = 15 # we use 15 images for each video.

        if self._image_set == 'train':
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr_VID_15frames_100vids_per_cat.txt')
            image_index = []
            if os.path.exists(image_set_file):
                f = open(image_set_file, 'r')
                data = f.readlines()
                # we need the prefix for the path.
                #prefix = 'data/imagenet/ILSVRC/Data/VID'
                for line in data:
                    if line != '':
                        image_index.append(line)
                f.close()
                return image_index

            # We use vid_indexes to contain all video folders for each category.
            vid_indexes = []
            for i in range(1,31):
                vid_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'train_' + str(i) + '.txt')
                with open(vid_set_file) as f:
                    tmp_index = [x.strip() for x in f.readlines()]
                    vtmp_index = []
                    for line in tmp_index:
                        line = line.split(' ')
                        # The list file given by ImageNet has some problem, we need to handle it by checking line item.
                        if line[1]=='1':
                            vtmp_index.append(os.path.dirname(line[0]))
                        else:
                            print(line)
                            raise 'Please check your file list.'

                    # Count file numbers for each video.
                    vid_indexes.append(vtmp_index)

            image_indexes =[]
            # First, we sample the videos for each category.
            for vid_idx in vid_indexes:
                vid_num = np.minimum(len(vid_idx), vid_num_per_cat)
                vid_idx = random.sample(vid_idx, vid_num)
                assert len(vid_idx)==vid_num and vid_num<=vid_num_per_cat
                # Second, we get images for each category.
                for vid_id in vid_idx:
                    flist = os.listdir(vid_id)
                    nfiles = len(flist)
                    ngap = np.float64(nfiles / (num_img_per_vid+1))
                    for iid in range(num_img_per_vid):
                        img_idx = int(np.round((iid+1)*ngap))
                        image_indexes.append(os.path.join(vid_id, '%06d'%(img_idx)))
            # Third, we shuffle the images for training.
            random.shuffle(image_indexes)
            print('Total number of video images are:%d'%(len(image_indexes)))
            num_lines = len(image_indexes)

            f = open(image_set_file, 'w')
            for line in image_indexes:
                f.write(line + '\n')
            f.close()
            return image_indexes
        else:
            image_set_file = os.path.join(self._data_path, 'ImageSets','VID', 'val.txt')
            vid_index = []
            with open(image_set_file) as f:
                for x in f.readlines():
                    line = x.strip().split(' ')
                    vid_index.append(self._data_path + '/Data/VID/val/' + line[0])
            return vid_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print(cache_file)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        print('len(self.image_index))', len(self.image_index))
        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        #filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')
        filename = index.replace('Data','Annotations')+'.xml'

        assert os.path.exists(filename),'%s'%(filename)
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)
        #filter the objects not in video synsets.
        #print('before:',num_objs)
        used_objs = []
        for id, obj in enumerate(objs):
            if str(get_data_from_tag(obj, "name")).lower().strip() in self._wnid_to_ind:
                used_objs.append(obj)
        objs = used_objs
        num_objs = len(objs)
        #print('after:', num_objs)
        ##########################################

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
