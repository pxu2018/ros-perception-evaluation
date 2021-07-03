import os
import pandas

from eval_utils import eval_object

    

def get_csv(file_name,detection = False):

    objects = [] 

    f = pandas.read_csv(file_name)
        
    for i in range(len(f)):

        frame = f['frame'][i]
        id = f['id'][i]
        type = f['type'][i]
        alpha = f['alpha'][i]
        (left,top) = (f['left'][i],f['top'][i])
        (right,bottom) = (f['right'][i],f['bottom'][i])
        (h,w,l) = (f['h'][i],f['w'][i],f['l'][i])
        (x,y,z) = (f['x'][i],f['y'][i],f['z'][i])
        rot = f['rotation_z'][i]
        (vx,vy) = (f['vx'][i],f['vy'][i])
        t = f['timestamp'][i]

        if detection == True:
            score = f['score'][i]
            objects.append(eval_object(frame,id,type,alpha,left,top,right,bottom,h,w,l,x,y,z,rot,vx,vy,t,score))
        else:
            objects.append(eval_object(frame,id,type,alpha,left,top,right,bottom,h,w,l,x,y,z,rot,vx,vy,t))

  
    return objects

