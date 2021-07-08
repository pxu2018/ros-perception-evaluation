import os
import pandas

from eval_utils import eval_object

    

def get_csv(file_name,detection = False,Camera = False, Radar = False):

    objects = [] 

    f = pandas.read_csv(file_name)

    if Camera == True:

        # Eliminate the objects outside the camera range of view filtering

        filter = f['left']!=-1
        f = f[filter]

    # elif Radar == True:

        # Filter based in range and angle

        # filter = 
        # f = f[filter]


    for i in range(len(f)):

        frame = f['frame'][i]
        id = f['id'][i]
        type = f['type'][i]
        alpha = f['alpha'][i]
        (left,top) = (f['left'][i],f['top'][i])
        (right,bottom) = (f['right'][i],f['bottom'][i])
        (h,w,l) = (f['size_z'][i],f['size_y'][i],f['size_x'][i])
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

