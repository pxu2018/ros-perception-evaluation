import os
import pandas

class eval_object():
    def __init__(self,alpha,left,top,right,bottom,h,w,l,rotation):
        self.bbox2D = [(left,top),(right,bottom)]
        self.dim = [h,w,l]
        self.alpha = alpha
        self.rotation = rotation

        

def get_csv(file_name):

    objects = [] 

    f = pandas.read_csv(file_name)
    print(f.head(1).frame)

    for i in range(len(f)):
        objects[i] = eval_object()