import math

from shapely.geometry import Polygon


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

    def __str__(self):
        return("Frame:{} Id:{} Loc:{}".format(self.frame,self.id,self.loc))

    def asociate_detected_object(self,object):

        #Only used in gt objects to save the correspondent detection

        if object.frame == self.frame:

            self.det_id = object.id
            self.det_bbox2d = object.bbox2d
            self.det_dim = object.dim
            self.det_loc = object.loc
            self.det_alpha = object.alpha
            self.det_rotation = object.rotation
  

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
 
        


def calc_distance(p1,p2):

    # p1 and p2 are touples of cordinates, in camera (x,z) in lidar (x,y)

    d = math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

    return d