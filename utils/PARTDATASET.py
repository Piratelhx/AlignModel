"""
PARTDATASET.py
Author: Hongxin Lin
Date: May 2019
"""
import h5py
import os
import json
import random 
import numpy as np
import sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider

class REGRESSDATASET(object):
    def __init__(self, DENSE_H5PY_FILE, PARTIAL_H5PY_FILE, JSON_FILE, BATCH_SIZE = 16, dense_npoints = 30000, partial_npoints = 2048, mode = "train", augument = False):
        self.dense_h5py_file = h5py.File(DENSE_H5PY_FILE,"r")
        self.partial_h5py_file = h5py.File(PARTIAL_H5PY_FILE, "r")
        self.anno = json.load(open(JSON_FILE,"r"))
        self.idx = 0
        self.length = len(self.anno)
        self.batch_size = BATCH_SIZE
        self.dense_npoints = dense_npoints
        self.partial_npoints = partial_npoints
        self.mode = mode
        self.augument = augument

        if self.mode == "train":
            random.shuffle(self.anno)
            print "======================== TRAIN MODE ========================"
        else:
            print "======================== TEST  MODE ========================"

        print "Dense h5py File have {} files !".format(len(self.dense_h5py_file.keys()))
        print "Partial h5py File have {} files !".format(len(self.partial_h5py_file.keys()))
        print "Anno File have {} files !".format(len(self.anno))
    
    def num_channel(self):
        return 3

    def next_batch(self):
        cur_partial_npoints = np.zeros((self.batch_size,self.partial_npoints,3),dtype = np.float32)
        cur_dense_npoints = np.zeros((self.batch_size, self.dense_npoints, 3),dtype = np.float32)
        cur_filename_list = []
        for b in range(self.batch_size):
            filename = self.anno[self.idx]
            point = self.partial_h5py_file[filename]
            point -= np.mean(point, axis = 0)
            if self.augument:
                point = provider.rotate_FR_single_data(point, rotate_angle = math.pi/6)
            cur_partial_npoints[b] = point
            
            dense_point = self.dense_h5py_file["face3_205-232-6_44_30_speckle_indoor-44-noglass-speckle-30cm-2019_01_16_12_19_14_286"] # Template Face#
            cur_dense_npoints[b] = dense_point - np.mean(dense_point, axis = 0)
            cur_filename_list.append(filename)
            self.idx += 1
            if(self.idx == self.length):
                self.idx = 0
                if self.mode == "train":
                    random.shuffle(self.anno)
        
        return cur_partial_npoints, cur_dense_npoints, cur_filename_list


    def has_next_batch(self):
        return self.idx < self.length
    
    def file_length(self):
        return self.length

if __name__ == "__main__":
    pass
    
  

    
        