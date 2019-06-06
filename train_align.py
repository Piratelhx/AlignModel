"""
TF model for partial point cloud regression. 
Author: Hongxin Lin
Date: May 2019
"""
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider 
from PARTDATASET import *
import json
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: dgcnn]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--full_npoints', type=int, default=4096, help='FULL Point Number [default: 4096]')
parser.add_argument('--part_npoints', type=int, default= 2048, help='part npoints')
parser.add_argument('--dense_npoints', type=int, default= 30000, help='dense npoints')
parser.add_argument('--max_epoch', type=int, default= 1000000, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default= 160000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--log_file', default='./log/', help='checkpoint dir [default: log]')
parser.add_argument('--restore', action='store_true', help='Whether to restore model')
parser.add_argument('--max_batch', type=int, default= 1000000, help='Epoch to run [default: 251]')
parser.add_argument('--fixed_learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--fixed', action='store_true', help='Whether to fixed lr')
parser.add_argument('--snapshot', type=int, default= 10000, help='Epoch to run [default: 251]')
parser.add_argument('--test_iter', type=int, default= 100000000000, help='Epoch to run [default: 251]')
parser.add_argument('--display_iter', type=int, default= 50, help='Epoch to run [default: 251]')
parser.add_argument('--weight', type=float, default=0.1, help='loss weight [default: 1]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
BEST_RES = 0

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
CHECKPOINT = FLAGS.log_file
RESTORE = FLAGS.restore
MAX_BATCH = FLAGS.max_batch
FIXED_LR = FLAGS.fixed_learning_rate
FIXED = FLAGS.fixed
Snapshot = FLAGS.snapshot
TEST_ITER = FLAGS.test_iter
Display_ITER = FLAGS.display_iter
FULL_NPOINTS = FLAGS.full_npoints
PART_NPOINTS = FLAGS.part_npoints
DENSE_NPOINTS = FLAGS.dense_npoints
Weight = FLAGS.weight

MODEL = importlib.import_module(FLAGS.model) # sym import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)


os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_align.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

DENSE_H5PY_FILE = "./data/real_data_30_50.hdf5"
H5PY_FILE = "./data/real_data_30_50.hdf5"
TRAIN_JSON_FILE = "./data/train.json"
EVAL_JSON_FILE = "./data/val.json"

TRAIN_DATASET = REGRESSDATASET(DENSE_H5PY_FILE, H5PY_FILE, TRAIN_JSON_FILE, BATCH_SIZE = BATCH_SIZE, dense_npoints = DENSE_NPOINTS, partial_npoints = PART_NPOINTS, mode = "train", augument = True)
EVAL_DATASET =  REGRESSDATASET(DENSE_H5PY_FILE, H5PY_FILE, EVAL_JSON_FILE, BATCH_SIZE = BATCH_SIZE, dense_npoints = DENSE_NPOINTS, partial_npoints = PART_NPOINTS, mode = "test", augument = False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate      
 

def get_fixed_learning_rate(lr):
    learning_rate = tf.constant(lr)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch * BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay



def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            partial_pc_pl, dense_pc_pl,  gt_normal, gt_bias = MODEL.placeholder_inputs(BATCH_SIZE, PART_NPOINTS, DENSE_NPOINTS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            rotate_matrix, translation, rotate_fullpoint = MODEL.get_model(partial_pc_pl, is_training_pl, bn_decay=bn_decay, name = 'sym')
            cd_loss, dists_forward, dists_backward = MODEL.get_loss(rotate_fullpoint, dense_pc_pl)

            loss = cd_loss

            log_string("--- Get training operator------")
            if not FIXED:
                learning_rate = get_learning_rate(batch)
            else:
                learning_rate = get_fixed_learning_rate(FIXED_LR)
            tf.summary.scalar('learning_rate', learning_rate)
           
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init,{is_training_pl: True})

        ops = {'partial_pc_pl': partial_pc_pl,
                'dense_pc_pl': dense_pc_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'learning_rate':learning_rate,
               'cd_loss': cd_loss,
               'rotate_fullpoint':rotate_fullpoint, 
               'dists_forward': dists_forward,
               'dists_backward': dists_backward,
               'rotate_matrix': rotate_matrix,
               'translation': translation}

        train_network(sess, ops, train_writer,saver)



def train_network(sess, ops, train_writer,saver):
    """ ops: dict mapping from string to tf ops """
    try:
        is_training = True
        BEST_RES = 1e20
        
        log_string(str(datetime.now()))
        
        if RESTORE:
            print "TESTING"
            saver.restore(sess, CHECKPOINT)
            BEST_RES  = eval_train_network(sess, ops)
            print("Model restored from %s"%CHECKPOINT)

        loss_sum = 0
        cd_loss_sum = 0
        start = time.time()
        while 1:
            cur_partial_npoints, cur_dense_npoints, cur_filename_list = TRAIN_DATASET.next_batch()
            feed_dict = {ops['partial_pc_pl']: cur_partial_npoints,
                        ops['dense_pc_pl']: cur_dense_npoints,
                        ops['is_training_pl']: is_training}
            summary, step, _, loss_val,  lr, cd_loss_val, rotate_pc, rot_matrix, translation_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['learning_rate'], ops['cd_loss'],  ops['rotate_fullpoint'], ops['rotate_matrix'], ops['translation']], feed_dict=feed_dict)
            
            loss_sum += loss_val
            
    
            cd_loss_sum += cd_loss_val
                           
            if  step % Display_ITER == 0 and step > 0:
                end = time.time()
                log_string('ITER: %d  Learning Rate: %f LOSS: %f  CD LOSS: %f Time: %f' \
                            %(step,lr, loss_sum / Display_ITER, cd_loss_sum / Display_ITER,  end - start))
                loss_sum = 0
                cd_loss_sum = 0
                start = time.time()
    
            if  step % TEST_ITER == 0 and step > 0:
                log_string("Begin Testing: ")
                res = eval_train_network(sess,ops)
                log_string("Ending Testing ")
                log_string("Result now on test : {} \n".format(res))
                log_string("Best Result util now is : {} \n".format(BEST_RES))
                     
            if step % Snapshot == 0 and step > 0:
                res = eval_train_network(sess,ops)
                if res < BEST_RES:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model_best.ckpt" ))
                    log_string("Model saved in file: %s" % save_path)
                    BEST_RES = res
                log_string("Best Result util now is : {} \n".format(BEST_RES))
            elif step > MAX_BATCH:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model%s.ckpt" % str(int(step))))
                log_string("Model saved in file: %s" % save_path)
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)       
     
    except KeyboardInterrupt:
        save_path = saver.save(sess, os.path.join(LOG_DIR, "model%s.ckpt" % str(int(step))))
        log_string("Model saved in file: %s" % save_path)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def eval_train_network(sess, ops):
    is_training = False
    loss_sum = 0
    dists_forward_sum = 0
    num_batches = EVAL_DATASET.file_length() //BATCH_SIZE
    total_seen = 0
    while total_seen <  EVAL_DATASET.file_length():
        cur_partial_npoints, cur_dense_npoints, cur_filename_list = EVAL_DATASET.next_batch()
        feed_dict = {ops['partial_pc_pl']: cur_partial_npoints,
                    ops['dense_pc_pl']: cur_dense_npoints,
                    ops['is_training_pl']: is_training}
        
        loss_val,  dists_forward_val, dists_backward_val, rotate_pc, rot_matrix, translation_val = sess.run([ops['loss'],  ops['dists_forward'], ops['dists_backward'], ops['rotate_fullpoint'], ops['rotate_matrix'], ops['translation']], feed_dict=feed_dict)
           
        loss_sum += loss_val
        dists_forward_sum += np.mean(dists_forward_val)

        total_seen += BATCH_SIZE

        if total_seen % 1024 == 0:
            print "Have Tested {} ".format(total_seen)
    
    log_string("EVAL LOSS: {}".format(loss_sum / float(num_batches)))
    log_string("EVAL DIST FORWARD LOSS: {}".format(dists_forward_sum  / float(num_batches)))

    return (dists_forward_sum) / float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
