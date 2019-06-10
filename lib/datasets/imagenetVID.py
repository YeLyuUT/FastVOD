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
try:
   import cPickle as pickle
   print('import cPickle')
except:
   import pickle
   print('import python pickle')
import random
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class imagenetVID(imdb):
    def __init__(self, image_set, devkit_path, data_path, gap=2):
        imdb.__init__(self, 'imagenetVID_' + image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = data_path
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))
        self._gap = gap
        self._classes = ('__background__',)
        self._wnid = (0,)

        for i in xrange(30):
            self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)

        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        self._image_ext = ['.JPEG']

        # structured_indexes hold indexes in structure of [cat..[vid..[imgs]]] for training, [vid..[imgs]] for testing.
        self._image_index, self._structured_indexes = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

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
        # print(index)
        # image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        image_path = index + self._image_ext[0]
        # print(image_path)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def get_vids_for_category(self, cat_ind):
        """
        Get video number of the category.
        The first category starts from cat_ind 1.
        """
        assert cat_ind > 0 and cat_ind <= 31, 'cat_ind should be in range from [1,30]'
        vid_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'train_' + str(cat_ind) + '.txt')
        tmp_index = []
        with open(vid_set_file) as f:
            tmp_index = [x.strip().split(' ')[0] for x in f.readlines()]
            for tmp in tmp_index:
                assert os.path.exists(self._data_path + '/Data/VID/train/' + tmp), ('Path not exists:%s') % (
                        self._data_path + '/Data/VID/train/' + tmp)
        return tmp_index

    def get_imgs_for_vid(self, cat_ind, vid_ind, isSorted=True):
        """
        Get images for video.
        """
        assert cat_ind > 0 and cat_ind <= 30, 'cat_ind should be in range from [1,30]'
        vid_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'train_' + str(cat_ind) + '.txt')
        tmp_index = []
        with open(vid_set_file) as f:
            tmp_index = [x.strip().split(' ')[0] for x in f.readlines()]
        assert vid_ind >= 0 and vid_ind < len(tmp_index)
        vid_dir = self._data_path + '/Data/VID/train/' + tmp_index[vid_ind]
        img_names = os.listdir(vid_dir)
        img_names = [os.path.join(vid_dir, name) for name in img_names]
        directory = os.path.dirname(img_names[0])
        for i in range(len(img_names)):
            assert os.path.exists(os.path.join(directory, '%06d.JPEG' % i)), 'The image %s is missing.' % (
                os.path.join(directory, '%06d.JPEG' % i))
        if isSorted:
            return [os.path.join(directory, '%06d.JPEG' % i) for i in range(len(img_names))]
        else:
            return img_names

    def get_anns_for_imgs(self, imgs):
        return [self._load_imagenet_annotation(img) for img in imgs]

    def get_trackids_set_from_anns(self, anns):
        trackids = set()
        for ann in anns:
            obj_trackids = ann['trackid']
            for o_tid in obj_trackids:
                trackids.add(o_tid)
        return trackids

    def _load_image_set_index(self):
        '''
        Load all the image index.
        '''
        if self._image_set == 'train':
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr_VID_TRA.pkl')
            image_index = []
            if os.path.exists(image_set_file):
                with open(image_set_file, 'r') as f:
                    '''
                    data = f.readlines()
                    for line in data:
                        if line != '':
                            image_index.append(line)
                    '''
                    image_index, structured_indexes = pickle.load(f)
                    return image_index, structured_indexes

            # We use vid_indexes to contain all video folders for each category.
            vid_indexes = []
            for i in range(1, 31):
                vid_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'train_' + str(i) + '.txt')
                with open(vid_set_file) as f:
                    tmp_index = [x.strip() for x in f.readlines()]
                    vtmp_index = []
                    for line in tmp_index:
                        line = line.split(' ')
                        if line[1] == '1':
                            vtmp_index.append(self._data_path + '/Data/VID/train/' + line[0])
                        else:
                            print(line)
                            raise 'Please check your file list.'

                    vid_indexes.append(vtmp_index)

            # list hold all video image indexes.
            image_indexes = []
            # [cat,...[vid,...[index...]]]
            structured_indexes = []
            idx_counter = 0
            for vid_idx in vid_indexes:
                cat_holder = []
                for vid_id in vid_idx:
                    vid_holder = []
                    flist = os.listdir(vid_id)
                    nfiles = len(flist)
                    for iid in range(self._gap // 2, nfiles, self._gap):
                        image_indexes.append(os.path.join(vid_id, '%06d' % (iid)))
                        vid_holder.append(idx_counter)
                        idx_counter += 1
                    cat_holder.append(vid_holder)
                structured_indexes.append(cat_holder)

            with open(image_set_file, 'w') as f:
                '''
                for line in image_indexes:
                        f.write(line + '\n')
                '''
                pickle.dump((image_indexes, structured_indexes), f, pickle.HIGHEST_PROTOCOL)
            return image_indexes, structured_indexes
        else:
            vid_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'val.txt')
            assert os.path.exists(vid_set_file), vid_set_file + ' does not exist.'
            image_indexes = []
            # [vid,...[index...]]
            structured_indexes = []
            last_dir = None
            tmp_holder = None
            i = 0
            with open(vid_set_file, 'r') as f:
                for x in f.readlines():
                    line = x.strip().split(' ')
                    dir_name = os.path.dirname(line[0])
                    if dir_name != last_dir:
                        if tmp_holder is not None:
                            structured_indexes.append(tmp_holder)
                        tmp_holder = []
                        last_dir = dir_name
                    image_indexes.append(self._data_path + '/Data/VID/val/' + line[0])
                    tmp_holder.append(i)
                    i += 1
                # Add the last vid.
                structured_indexes.append(tmp_holder)

            print('Total number of videos are: %d.' % (len(structured_indexes)))
            print('Total number of video images are: %d.' % (len(image_indexes)))
            return image_indexes, structured_indexes

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        The roidb is list of list of shape (cat_num, vid_num, img_num)
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print(cache_file)

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = []
        for img_idx in self.image_index:
            gt_roidb.append(self._load_imagenet_annotation(img_idx))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        # filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')
        filename = index.replace('Data', 'Annotations').replace('.JPEG', '') + '.xml'

        assert os.path.exists(filename), '%s' % (filename)

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)
        # filter the objects not in video synsets.
        # print('before:',num_objs)
        used_objs = []
        for id, obj in enumerate(objs):
            if str(get_data_from_tag(obj, "name")).lower().strip() in self._wnid_to_ind:
                used_objs.append(obj)
        objs = used_objs
        num_objs = len(objs)
        # print('after:', num_objs)
        ##########################################

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        trackid = np.zeros((num_objs), dtype=np.int32)

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
            trackid[ix] = int(get_data_from_tag(obj, 'trackid'))

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'trackid': trackid,
                'flipped': False}

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
