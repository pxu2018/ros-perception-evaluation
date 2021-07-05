import io
import time 
import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import ptp


from file_utils import get_csv
from eval_utils import info_classes, iou_2d, iou_dist_3d, get_status, calc_vel_error

CLASS_LIST = ["Unknown", "Unknown_Small","Unknown_Medium","Unknown_Big","Pedestrian", "Bike","Car", "Truck","Motorcycle", "Other_Vehicle","Barrier", "Sign"]

FILE_NAME = "/prueba_det.csv"
GT_FILE = "/prueba_kitti.csv"

DIST_THRESHOLD = 1
IOU2D_THRESHOLD = 0.5



def main():

    elements = get_csv(os.getcwd()+ FILE_NAME,detection=True)
    ground_tr = get_csv(os.getcwd()+ GT_FILE)
    
    id = 0

    # Iterations with gt instead of detections (podría ser que la detección mas cercana de 2 gt sea la misma, para esos casos se configura el threshold de distancia al gt)

    for gt in ground_tr:

        print(id)
        id = id +1

        d_min = 50
        associate_index = None

        for i in range(len(elements)):

            el = elements[i]

            if el.timestamp == gt.timestamp: ##Falta asociar con timestamp de ROS

                iou2d = iou_2d(gt.bbox2d,el.bbox2d)
                iou3d, distance = iou_dist_3d(gt,el)

                if distance < DIST_THRESHOLD and iou2d > IOU2D_THRESHOLD and distance < d_min: 
                    
                    associate_index = i
                    d_min = distance

                    vel_error = calc_vel_error(gt.v,el.v)
                    asoc_iou = iou3d
                    asoc_d = distance


        if associate_index is not None and elements[associate_index].status == 0:
            gt.asociate_object(elements[associate_index],asoc_iou,asoc_d,vel_error)
            elements[associate_index].asociate_object(gt,asoc_iou,asoc_d,vel_error)

            
    det_status = get_status(elements)
    det_status = np.flip(det_status[:,det_status[1,:].argsort()],axis = 1)
  
    tp_fn = len(ground_tr)
    acc_TP = 0
    acc_FP = 0
    precision = []
    recall = []
    classes_data = info_classes()
    
    for i in range(np.shape(det_status)[1]):

        name = det_status[2,i]

        # Update number of detections and statistics of the detected class

        value = np.array(getattr(classes_data,name))
        new_val = (value + np.array([0,1,0,0])).tolist()
        setattr(classes_data,name,new_val)

        # If the detection is asociated with a gt, it is a TP (update the value in the data class), if not it is a FP

        if int(det_status[0,i]) == 1:

            acc_TP = acc_TP + 1
        
            value = getattr(classes_data,name)
            new_val = (value + np.array([1,0,float(det_status[3,i]),float(det_status[5,i])])).tolist()
            setattr(classes_data,name,new_val)

        else:
            acc_FP = acc_FP + 1

        precision.append(acc_TP/(acc_TP+acc_FP))
        recall.append(acc_TP/tp_fn)


    value_mat,mAP = classes_data.calculate_AP_mAP()
    
    plt.plot(recall, precision)
    plt.title('Precision-recall curve')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

    print("RESULTS"+os.linesep)
    print("Total:"+os.linesep+"TP:{}  FP:{}  FN:{}".format(acc_TP,acc_FP,(tp_fn-acc_TP)))
    print("Precision values:{}  Recall values:{}".format(precision,recall)+os.linesep)
    print("Classes" + os.linesep + "mAP:{}".format(mAP)+ os.linesep +"AP | mIoU | mAVE for each class:"+os.linesep+"{}".format(value_mat)+os.linesep)
    
      










if __name__ == "__main__":
    main()