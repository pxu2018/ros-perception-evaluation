import os
import pandas

from eval_utils import eval_object

x_camera = 0.41
y_camera = 0
z_camera = 1.64

x_lidar = 0
y_lidar = 0
z_lidar = 1.95

x_radar = 2
y_radar = 0
z_radar = 0.65


def get_csv(file_name, detection=False, Camera=False, Radar=False, Lidar=False, conf_threshold=-1):
    objects = []
    f = pandas.read_csv(file_name)
    if conf_threshold != -1:
        # Filtering by confidence
        filter = f['score'] > conf_threshold
        f = f[filter]
        f = f.reset_index()

    x_off = 0
    y_off = 0
    z_off = 0

    if Camera == True:
        # Eliminate the objects outside the camera range of view filtering
        filter = f['left'] != -1
        f = f[filter]
        f = f.reset_index()
        x_off = x_camera
        y_off = y_camera
        z_off = z_camera
    elif Radar == True:
        # Filter based in range and angle
        # filter = 
        # f = f[filter]
        x_off = x_radar
        y_off = y_radar
        z_off = z_radar
    elif Lidar == True:
        x_off = x_lidar
        y_off = y_lidar
        z_off = z_lidar

    for i in range(len(f)):
        frame = f['frame'][i]
        id = f['id'][i]
        type = f['type'][i]
        alpha = f['alpha'][i]
        (left, top) = (f['left'][i], f['top'][i])
        (right, bottom) = (f['right'][i], f['bottom'][i])
        (h, w, l) = (f['h'][i], f['w'][i], f['l'][i])
        (x, y, z) = (f['x'][i] + x_off, f['y'][i] + y_off, f['z'][i] + (z_off / 2))
        rot = f['rotation_z'][i]
        (vx, vy) = (f['vx'][i], f['vy'][i])
        t = f['timestamp'][i]
        if detection == True:
            score = f['score'][i]
            objects.append(
                eval_object(frame, id, type, alpha, left, top, right, bottom, h, w, l, x, y, z, rot, vx, vy, t, score))
        else:
            objects.append(
                eval_object(frame, id, type, alpha, left, top, right, bottom, h, w, l, x, y, z, rot, vx, vy, t))

    return objects


def get_gt_stats(file_name, camera=False, obj_class='Car'):
    f = pandas.read_csv(file_name)
    if camera == True:
        filter = f['left'] != -1
        f = f[filter]
        f = f.reset_index()

    filter = f['type'] == obj_class
    f = f[filter]
    f = f.reset_index()
    return len(f)
