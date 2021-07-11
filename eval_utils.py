import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from geometric_utils import create_3d_bbox, calc_distance_2d


class eval_object():
    def __init__(self,frame,id,type,alpha,left,top,right,bottom,h,w,l,x,y,z,rotation,vx,vy,timestamp,score = None):
        self.frame = frame
        self.id = id
        self.type = type
        self.bbox2d = [(left,top),(right,bottom)]
        self.dim = [h,w,l]
        self.loc = [x,y,z]
        self.alpha = alpha
        self.rotation = rotation
        self.v = (vx,vy)
        self.timestamp = timestamp
        self.distance = 100
        self.status = 0
        self.score = score
        self.iou = 0
        self.d = 0
        self.v_err = 0

    def __str__(self):
        return("Frame:{} Id:{} Loc:{} Status:{}".format(self.frame,self.id,self.loc,self.status))

    def asociate_object(self,object,iou,d,v_err):

        # We use the minimum distance to asociate the ground-truth with a detection

        self.det_id = object.id
        self.det_bbox2d = object.bbox2d
        self.det_dim = object.dim
        self.det_loc = object.loc
        self.det_alpha = object.alpha
        self.det_rotation = object.rotation
        self.iou = iou
        self.d = d
        self.v_err = v_err
        self.set_status()
 
  
    def set_status(self):

        # 0 if not associated (FN or FP), 1 if TP

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
    
    return iou
 

def iou_dist_3d(obj1,obj2):

    # Both obj1 and obj2 are eval_object class elements

    box3d_1 = create_3d_bbox(obj1)
    box3d_2 = create_3d_bbox(obj2)
    bird_poly1 = Polygon([(box3d_1[0,0],box3d_1[1,0]),(box3d_1[0,1],box3d_1[1,1]),(box3d_1[0,2],box3d_1[1,2]),(box3d_1[0,3],box3d_1[1,3])])
    bird_poly2 = Polygon([(box3d_2[0,0],box3d_2[1,0]),(box3d_2[0,1],box3d_2[1,1]),(box3d_2[0,2],box3d_2[1,2]),(box3d_2[0,3],box3d_2[1,3])])
 
    # x,y = bird_poly1.exterior.xy
    # plt.plot(y,x)
    # x,y = bird_poly2.exterior.xy
    # plt.plot(y,x)
    # plt.show()


    #bird_inters_poly = bird_poly1.intersection(bird_poly2).exterior.coords[:-1]
    bird_inters = bird_poly1.intersection(bird_poly2).area
    
    
    low = max([box3d_1[2,0],box3d_2[2,0]])
    top = min([box3d_1[2,4],box3d_2[2,4]])
    h = top-low
    

    inter_volume = bird_inters*h

    vol1 = abs(box3d_1[2,0]-box3d_1[2,4]) * bird_poly1.area
    vol2 = abs(box3d_2[2,0]-box3d_2[2,4]) * bird_poly2.area

    iou = inter_volume/(vol1+vol2-inter_volume)
    print(obj1.loc,obj2.loc)

    d = calc_distance_2d((obj1.loc[0],obj1.loc[1]),(obj2.loc[0],obj2.loc[1]))
 
    return iou,d


def get_status(list):

    # Returns status and score of a detection list
    
    f1 = lambda x: x.status
    vectorized_f = np.vectorize(f1)

    f2 = lambda x: x.score
    vectorized_f2 = np.vectorize(f2)

    f3 = lambda x: x.type
    vectorized_f3 = np.vectorize(f3)

    f4 = lambda x: x.iou
    vectorized_f4 = np.vectorize(f4)
    
    f5 = lambda x: x.d
    vectorized_f5 = np.vectorize(f5)

    f6 = lambda x: x.v_err
    vectorized_f6 = np.vectorize(f6)
   

    return np.vstack((vectorized_f(list),vectorized_f2(list),vectorized_f3(list),vectorized_f4(list),vectorized_f5(list),vectorized_f6(list)))



class info_classes():

    # For each class we save a list of [TP , totatl number of detections, accIoU, accVE] to calculate AP (FP = total - TP)

    def __init__(self):
        self.names = ["Unknown", "Unknown_Small","Unknown_Medium","Unknown_Big","Pedestrian", "Bike","Car", "Truck","Motorcycle", "Other_Vehicle","Barrier", "Sign"]
        self.Unknown = [0,0,0,0]
        self.Unknown_Small = [0,0,0,0]
        self.Unknown_Medium = [0,0,0,0]
        self.Unknown_Big = [0,0,0,0]
        self.Pedestrian = [0,0,0,0]
        self.Bike = [0,0,0,0]
        self.Car = [0,0,0,0]
        self.Truck = [0,0,0,0]
        self.Motorcycle = [0,0,0,0]
        self.Other_Vehicle = [0,0,0,0]
        self.Barrier = [0,0,0,0]
        self.Sign = [0,0,0,0]

    def calculate_AP_mAP(self):
        
        AP = []
        IoU = []
        VE = []

        for name in self.names:
    
            val = getattr(self,name)
            TP = val[0]
            total_det = val[1]
            FP = total_det - TP

            accIoU = float(val[2])
            accVE = float(val[3])
        
            if (TP+FP) != 0:
                AP.append(float(TP/(TP+FP)))   
            else:
                AP.append(float(0)) 

            if TP != 0:
                IoU.append(float(accIoU/TP))
                VE.append(float(accVE/TP))
            else:
                IoU.append(float(0))
                VE.append(float(0))

        
        v = np.vectorize(lambda x: x > 0)
        if sum(v(AP)) != 0:
            mAP = sum(AP)/sum(v(AP))
        else:
            mAP = 0

        return np.transpose(np.vstack((self.names,AP,IoU,VE))),mAP


        

def calc_vel_error(vel1,vel2):

    # Vel are touples (vx,vy)

    a = math.pow(np.abs(vel1[0]-vel2[0]),2)
    b = math.pow(np.abs(vel1[1]-vel2[1]),2)

    return math.sqrt(a+b)






        




