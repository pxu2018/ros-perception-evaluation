import io
import time 
import os
import numpy as np


from file_utils import get_csv
from eval_utils import info_classes, iou_2d, iou_dist_3d, get_status

CLASS_LIST = ["Unknown", "Unknown_Small","Unknown_Medium","Unknown_Big","Pedestrian", "Bike","Car", "Truck","Motorcycle", "Other_Vehicle","Barrier", "Sign"]

FILE_NAME = "/UwU.csv"
GT_FILE = "/UwU_gt.csv"

DIST_THRESHOLD = 1
IOU2D_THRESHOLD = 0.5



def main():

    elements = get_csv(os.getcwd()+ FILE_NAME,detection=True)
    ground_tr = get_csv(os.getcwd()+ GT_FILE)
    

    # Iterations with gt instead of detections (podría ser que la detección mas cercana de 2 gt sea la misma, para esos casos se configura el threshold de distancia al gt)

    for gt in ground_tr:

        d_min = 50
        associate_index = None

        for i in range(len(elements)):

            el = elements[i]

            iou2d, poly = iou_2d(gt.bbox2d,el.bbox2d)
            iou3d, distance = iou_dist_3d(gt,el)

            if distance < DIST_THRESHOLD and iou2d > IOU2D_THRESHOLD and distance < d_min: ##Falta asociar con timestamp
                
                associate_index = i
                d_min = distance

        if associate_index is not None and elements[associate_index].status == 0:
            gt.asociate_object(elements[associate_index])
            elements[associate_index].asociate_object(gt)

            
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

        # Update number of detections of the detected class

        value = np.array(getattr(classes_data,name))
        new_val = (value + np.array([0,1])).tolist()
        setattr(classes_data,name,new_val)

        # If the detection is asociated with a gt, it is a TP (update the value in the data class), if not it is a FP

        if int(det_status[0,i]) == 1:

            acc_TP = acc_TP + 1

            value = getattr(classes_data,name)
            new_val = (value + np.array([1,0])).tolist()
            setattr(classes_data,name,new_val)

        else:
            acc_FP = acc_FP + 1

        precision.append(acc_TP/(acc_TP+acc_FP))
        recall.append(acc_TP/tp_fn)


    AP_list,mAP = classes_data.calculate_AP_mAP()
    

    print("RESULTS"+os.linesep)
    print("Total:"+os.linesep+"TP:{}  FP:{}  FN:{}".format(acc_TP,acc_FP,(tp_fn-acc_TP)))
    print("Precision values:{}  Recall values:{}".format(precision,recall)+os.linesep)
    print("Classes" + os.linesep + "mAP:{}".format(mAP)+ os.linesep + "{}".format(AP_list))

      










if __name__ == "__main__":
    main()