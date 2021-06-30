import math
import numpy as np

from shapely.geometry import Polygon

from geometric_utils import create_3d_bbox, calc_distance_2d


class eval_object():
    def __init__(self,frame,id,alpha,left,top,right,bottom,h,w,l,x,y,z,rotation,timestamp):
        self.frame = frame
        self.id = id
        self.bbox2d = [(left,top),(right,bottom)]
        self.dim = [h,w,l]
        self.loc = [x,y,z]
        self.alpha = alpha
        self.rotation = rotation
        self.timestamp = timestamp
        self.distance = 100
        self.status = 0

    def __str__(self):
        return("Frame:{} Id:{} Loc:{} Status:{}".format(self.frame,self.id,self.loc,self.status))

    def asociate_detected_object(self,object,dis):

        # Only used in gt objects to save the correspondent detection
        # We use the minimum distance to asociate the ground-truth with a detection

        if dis < self.distance: 
            self.det_id = object.id
            self.det_bbox2d = object.bbox2d
            self.det_dim = object.dim
            self.det_loc = object.loc
            self.det_alpha = object.alpha
            self.det_rotation = object.rotation
            self.distance = dis
            self.set_status()
            return 1
        else:
            return 0
  
    def set_status(self):
        self.status = 1
  

def iou_2d(box1,box2):

    # Both boxes are a list of two touples [(left,top),(right,bottom)]

    h = [box1[1][1]-box1[0][1],box2[1][1]-box2[0][1]]
    w = [box1[1][0]-box1[0][0],box2[1][0]-box2[0][0]]

    a1 = box1[0]
    b1 = (box1[0][0],box1[0][1]+h[0])
    c1 = box1[1]
    d1 = (box1[1][0],box1[1][1]-h[0])

    a2 = box2[0]
    b2 = (box2[0][0],box2[0][1]+h[0])
    c2 = box2[1]
    d2 = (box2[1][0],box2[1][1]-h[0])

    poly1 = Polygon([a1,b1,c1,d1])
    poly2 = Polygon([a2,b2,c2,d2])

    iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    intersect_polygon = poly1.intersection(poly2).exterior.coords[:-1]
    return iou, intersect_polygon
 

def iou_dist_3d(obj1,obj2):

    # Both obj1 and obj2 are eval_object class elements

    box3d_1 = create_3d_bbox(obj1)
    box3d_2 = create_3d_bbox(obj2)
    bird_poly1 = Polygon([(box3d_1[0,0],box3d_1[1,0]),(box3d_1[0,1],box3d_1[1,1]),(box3d_1[0,2],box3d_1[1,2]),(box3d_1[0,3],box3d_1[1,3])])
    bird_poly2 = Polygon([(box3d_2[0,0],box3d_2[1,0]),(box3d_2[0,1],box3d_2[1,1]),(box3d_2[0,2],box3d_2[1,2]),(box3d_2[0,3],box3d_2[1,3])])

    #bird_inters_poly = bird_poly1.intersection(bird_poly2).exterior.coords[:-1]
    bird_inters = bird_poly1.intersection(bird_poly2).area
    
    low = max([box3d_1[2,0],box3d_2[2,0]])
    top = min([box3d_1[2,4],box3d_2[2,4]])
    h = top-low

    inter_volume = bird_inters*h

    vol1 = abs(box3d_1[2,0]-box3d_1[2,4]) * bird_poly1.area
    vol2 = abs(box3d_2[2,0]-box3d_2[2,4]) * bird_poly2.area

    iou = inter_volume/(vol1+vol2-inter_volume)

    d = calc_distance_2d((obj1.loc[0],obj1.loc[1]),(obj2.loc[0],obj2.loc[1]))
 
    return iou,d
    

