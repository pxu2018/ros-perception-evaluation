import io
import time 
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import pandas as pd

from numpy.core.fromnumeric import ptp, size


from file_utils import get_csv
from eval_utils import info_classes, iou_2d, iou_dist_3d, get_status, calc_vel_error

#CLASS_LIST = ["Unknown", "Unknown_Small","Unknown_Medium","Unknown_Big","Pedestrian", "Bike","Car", "Truck","Motorcycle", "Other_Vehicle","Barrier", "Sign"]
CLASS_LIST = ["Pedestrian","Car"]

# FILE_NAME = "/home/robesafe/t4ac_ws/src/t4ac_carla_simulator/ad_devkit/databases/perception/lidar/detections.csv"
# GT_FILE = "/home/robesafe/t4ac_ws/src/t4ac_carla_simulator/ad_devkit/databases/perception/groundtruth.csv"

FILE_NAME = "/home/miguel/TFG/perception-evaluation/csv/ROS_det.csv"
GT_FILE = "/home/miguel/TFG/perception-evaluation/csv/groundtruth.csv"

DIST_THRESHOLD = 2
CONF_THRESHOLD = np.arange(0,1.1,0.05)
IOU3D_THRESHOLD = 0.2
TIMESTAMP_RANGE = 0.05/2

# Only one sensor.

LIDAR_SENSOR = True
CAMERA_SENSOR = False
RADAR_SENSOR = False



def main():

    # elements = get_csv(FILE_NAME,detection = True,Camera = CAMERA_SENSOR,Radar = RADAR_SENSOR,Lidar=LIDAR_SENSOR)
    ground_tr = get_csv(GT_FILE) # Falta filtrar Radar en caso de usarse

    # Saber que sensor se esta evaluando
    # Si es LiDAR o radar se mantienen todas, si es camara hay que filtrar los elementos que tengan -1 en las bbox 2d

    acc_AP = 0


    for type in CLASS_LIST:
        precision = []
        recall = []
        mIoU = 0
        mAVE = 0
        f = pd.read_csv(GT_FILE)
        filter = f['type']==type
        f = f[filter]
        tp_fn = len(f)
        print("Evaluating class {}".format(type))

        for threshold in CONF_THRESHOLD:
            
            elements = get_csv(FILE_NAME,detection = True,Camera = CAMERA_SENSOR,Radar = RADAR_SENSOR,Lidar=LIDAR_SENSOR,conf_threshold=threshold)
            print("Number of elements with {} threshold: {}".format(round(threshold,2),len(elements)))

            # Iterations with gt instead of detections 
            for gt in ground_tr:

                d_min = 50
                associate_index = None

                for i in range(len(elements)):

                    el = elements[i]

                    # if el.timestamp > (gt.timestamp - TIMESTAMP_RANGE) and el.timestamp < (gt.timestamp + TIMESTAMP_RANGE): # ROS timestamp association
                    if round(el.timestamp,4) == round(gt.timestamp,4) : # ROS timestamp association
                        if CAMERA_SENSOR == True:
                            iou2d = iou_2d(gt.bbox2d,el.bbox2d)

                        iou3d, distance = iou_dist_3d(gt,el)
                        # print("Time:{} {}".format(el.timestamp,gt.timestamp))
                        # print("Distance:{}".format(distance))

                        if distance <= DIST_THRESHOLD and iou3d >= IOU3D_THRESHOLD and distance < d_min: 
                            associate_index = i
                            d_min = distance

                            vel_error = calc_vel_error(gt.v,el.v)
                            asoc_iou = iou3d
                            asoc_d = distance

                if associate_index is not None and elements[associate_index].status == 0:
                    #gt.asociate_object(elements[associate_index],asoc_iou,asoc_d,vel_error)
                    elements[associate_index].asociate_object(gt,asoc_iou,asoc_d,vel_error)
                    
                    

            if len(elements) != 0:  

                det_status = get_status(elements)
                det_status = np.flip(det_status[:,det_status[1,:].argsort()],axis = 1)

                classes_data = info_classes()

                for i in range(np.shape(det_status)[1]):

                    name = det_status[2,i]

                    if name == type:

                        # Update number of detections and statistics of the detected class

                        value = np.array(getattr(classes_data,name))
                        new_val = (value + np.array([0,1,0,0])).tolist()
                        setattr(classes_data,name,new_val)

                        # If the detection is asociated with a gt, it is a TP (update the value in the data class), if not it is a FP

                        if int(det_status[0,i]) == 1:
                        
                            value = getattr(classes_data,name)
                            new_val = (value + np.array([1,0,float(det_status[3,i]),float(det_status[5,i])])).tolist()
                            setattr(classes_data,name,new_val)



            P,R,IoU,AVE = classes_data.calculate_metrics(tp_fn,type)
            print(P)

            precision.append(P)
            recall.append(R)

            if threshold == 0.5:
                mIoU = IoU
                mAVE = AVE



        # print(precision)
        # print(recall)

        #AP = np.trapz(precision,recall)
        AP = auc(recall,precision)
        acc_AP = acc_AP + AP

        if AP != 0:
            plt.figure()
            plt.ylim(0,1)
            plt.xlim(0,1)
            plt.plot(recall, precision)
            plt.title('Precision-recall curve for {} class'.format(type))
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.show()
            
            print("{}".format(type)+os.linesep+"AP:{} mIoU:{} mAVE:{}".format(AP,mIoU,mAVE)+os.linesep)
            #print("Precision values:{}  Recall values:{}".format(precision,recall)+os.linesep)
            

    print("Total:"+os.linesep+"mAP:{}".format((acc_AP/1)))


if __name__ == "__main__":
    main()
    