""" 
Symmetry Model For Partial Point Cloud Restruction
Using Chamfer's distance loss and Frontial Loss
Author: Hongxin Lin
Date: May 2019
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
import h5py
import json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/nn_distance'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
import tf_nndistance
from tf_grouping import query_ball_point, group_point,knn_point
import tf_util
from transform_nets import input_transform_net
import provider


def quaternion2rotmatrix(quaternion):
    """
    Convert quaternion To the corresponding rotation matrix.
    Input:
        quaternion: B x 4
    Output:
        rot matrix: B x 3 x 3
    """

    def diag(a, b): 
        return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

    def tr_add(a, b, c, d):  
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  
        return 2 * a * b - 2 * c * d

    w, x, y, z = tf.unstack(quaternion, num = 4, axis = -1)
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
            [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
            [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
    
    return tf.stack([tf.stack(m[i], axis=-1) for i in range(3)], axis=-2)


def placeholder_inputs(batch_size, part_npoints, dense_npoints):
    """
    placeholder for input
    """
    partial_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, part_npoints, 3))
    dense_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, dense_npoints, 3))
    normal = tf.placeholder(tf.float32, shape = (batch_size,3))
    bias = tf.placeholder(tf.float32, shape = (batch_size, 1))
    
    return partial_pointclouds_pl, dense_pointclouds_pl, normal, bias


def get_model(point_cloud, is_training, reuse = False, bn_decay=None, name = 'sym'):
    """ 
    Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        plane parameters
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    input_image = tf.expand_dims(point_cloud, -1)
    
    net = tf_util.conv2d(input_image, 64, [1,point_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope= name + 'conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope= name + 'conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope= name + 'conv3', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [net.shape[1],1],
                                     padding='VALID', scope= name + 'maxpool')

    feature_net = tf.reshape(global_feat, [batch_size, -1])
    """ 
    Transform Matrix Regression
    """
    net = tf_util.fully_connected(feature_net, 64, bn=True, is_training=is_training,scope = name + 'fc4', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope = name + 'dp3')
    net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training,scope = name + 'fc5', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope = name + 'dp4')
   
    net = tf_util.fully_connected(net, 16, bn=True, is_training=is_training,scope = name + 'fc6', bn_decay=bn_decay)
    quaternion = tf_util.fully_connected(net, 4, activation_fn=None, scope= name + 'fc7')
    quaternion =  tf.nn.l2_normalize(quaternion, dim = 1)
    rot_matrix = quaternion2rotmatrix(quaternion)

    """
    Translation Regression
    """
    net = tf_util.fully_connected(feature_net, 64, bn=True, is_training=is_training,scope = name + 'fc8', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope = name + 'dp5')
    net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training,scope = name + 'fc9', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope = name + 'dp6')
   
    net = tf_util.fully_connected(net, 16, bn=True, is_training=is_training,scope = name + 'fc10', bn_decay=bn_decay)
    translation = tf_util.fully_connected(net, 3, activation_fn=None, scope= name + 'fc11')
    
    """
    Partial Completion By Using Reflection Matrix And Rot Matrix
    """
    rotate_fullpoint = tf.matmul(point_cloud, rot_matrix)

    rotate_fullpoint += tf.expand_dims(translation, axis = 1)

    return rot_matrix, translation, rotate_fullpoint


def get_loss(fullpoint, dense_pc_pl):
    dists_forward, _, dists_backward,_ = tf_nndistance.nn_distance(fullpoint, dense_pc_pl)
    cd_loss = tf.reduce_mean(dists_forward) + tf.reduce_mean(dists_backward)
                                                                                                                                                                                                                                                                                                                                                                                                                         
    return cd_loss, dists_forward, dists_backward

    

       
        
    
    
 