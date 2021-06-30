import math
import numpy as np

def calc_distance_2d(p1,p2):

    # p1 and p2 are touples of cordinates, in camera (x,z) in lidar (x,y)

    d = math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

    return d

def rotation_mat(angle):

    # Creates a rotation matrix around the height axis (y in camera and z in lidar)

    sin = math.sin(angle)
    cos = math.cos(angle)
    rot = np.vstack(([cos,-sin,0],[sin,cos,0],[0,0,1]))

    return rot

def create_3d_bbox(object):

    # Creates a 3d bbox in real cordinates. Obj is a eval_object class element

    rot = rotation_mat(object.rotation)
    
    (h,w,l) = (object.dim[0],object.dim[1],object.dim[2])
    (x,y,z) = (object.loc[0],object.loc[1],object.loc[2])

    cube = np.vstack(([-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2],[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],[-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]))
    offset = np.vstack((np.full((1,8),x),np.full((1,8),y),np.full((1,8),z)))  

    box = np.matmul(rot,cube) + offset

    return box

