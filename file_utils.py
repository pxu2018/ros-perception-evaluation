import os
import pandas

from eval_utils import eval_object

    

def get_csv(file_name):

    objects = [] 

    f = pandas.read_csv(file_name)
        
    for i in range(len(f)):

        frame = f['frame'][i]
        id = f['id'][i]
        alpha = f['alpha'][i]
        (left,top) = (f['left'][i],f['top'][i])
        (right,bottom) = (f['right'][i],f['bottom'][i])
        (h,w,l) = (f['h'][i],f['w'][i],f['l'][i])
        (x,y,z) = (f['x'][i],f['y'][i],f['z'][i])
        rot = f['rotation_z'][i]
        t = f['timestamp'][i]

        objects.append(eval_object(frame,id,alpha,left,top,right,bottom,h,w,l,x,y,z,rot,t))

  
    return objects

