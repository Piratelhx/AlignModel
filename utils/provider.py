import numpy as np
import random
import math
import h5py
import time
import os

def random_vector():
    u = random.random()
    v = random.random()
    theta = 2*math.pi*u
    phi = np.arccos(2*v - 1)
    x = math.sin(theta)*math.sin(phi)
    y = math.cos(theta)*math.sin(phi)
    z = math.cos(phi)
    return x,y,z

def quaternion2rotmatrix(quaternion):
    RT_transform_matrix = np.array(np.zeros((3,3) , dtype = np.float32))
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    s = quaternion[0]
    RT_transform_matrix[0,0] = 1 - 2*(y**2 + z**2)      
    RT_transform_matrix[1,1] = 1 - 2*(x**2 + z**2)      
    RT_transform_matrix[2,2] = 1 - 2*(x**2 + y**2)      
    RT_transform_matrix[0,1] = 2*(x*y - s*z)
    RT_transform_matrix[0,2] = 2*(x*z + s*y)
    RT_transform_matrix[1,0] = 2*(x*y + s*z)
    RT_transform_matrix[1,2] = 2*(y*z - s*x)
    RT_transform_matrix[2,0] = 2*(x*z - s*y)
    RT_transform_matrix[2,1] = 2*(y*z + s*x)
    return RT_transform_matrix

def rotate_FR_single_data(point,rotate_angle = math.pi/6):
    """ Randomly perturb the point clouds by small rotations
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, rotated batch of point clouds
    """
    ux,uy,uz = random_vector()
    theta = random.uniform(-rotate_angle,rotate_angle)
    quaternion = np.zeros(4)
    quaternion[0] = math.cos(theta/2)
    quaternion[1],quaternion[2],quaternion[3] = math.sin(theta/2) * ux,math.sin(theta/2) * uy,math.sin(theta/2) * uz
    R = quaternion2rotmatrix(quaternion)
    rotated_data = np.zeros(point.shape, dtype=np.float32)
    rotated_data = np.dot(point.reshape((-1, 3)),np.transpose(R))
    return rotated_data

def rotate_single_data(point,rotate_angle = math.pi/6):
    """ Randomly perturb the point clouds by small rotations
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, rotated batch of point clouds
    """
    ux,uy,uz = random_vector()
    theta = random.uniform(-rotate_angle,rotate_angle)
    quaternion = np.zeros(4)
    quaternion[0] = math.cos(theta/2)
    quaternion[1],quaternion[2],quaternion[3] = math.sin(theta/2) * ux,math.sin(theta/2) * uy,math.sin(theta/2) * uz
    R = quaternion2rotmatrix(quaternion)
    rotated_data = np.zeros(point.shape, dtype=np.float32)
    rotated_data = np.dot(point.reshape((-1, 3)),np.transpose(R))
    return rotated_data, R





    
    

