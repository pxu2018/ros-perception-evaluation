import io
import time 
import os
import numpy as np

from file_utils import get_csv
from eval_utils import iou_2d, iou_dist_3d


FILE_NAME = "/UwU.csv"
GT_FILE = "/UwU_gt.csv"

DIST_THRESHOLD = 1
IOU2D_THRESHOLD = 0.5



def main():

    elements = get_csv(os.getcwd()+ FILE_NAME)
    ground_tr = get_csv(os.getcwd()+ GT_FILE)
    
    # Iterations with gt instead of detections

    for gt in ground_tr:

        print("Gt:{}".format(gt)+os.linesep)

        for el in elements:

            iou2d, poly = iou_2d(gt.bbox2d,el.bbox2d)
            iou3d, distance = iou_dist_3d(gt,el)

            if distance < DIST_THRESHOLD and iou2d > IOU2D_THRESHOLD: ##Falta asociar con timestamp

                i = gt.asociate_detected_object(el,distance)


    #Arreglar
    #El numero de status 1 en gt debe ser igual que el de detections, status en detetcions no es necesario

    # Status de cada objeto de forma paralela

    f = lambda x: x.status
    vectorized_f = np.vectorize(f)

    status_vector = np.sum(vectorized_f(ground_tr))
    print(gt_status)

 

    for gt in ground_tr:

        print("Gt:{}".format(gt)+os.linesep)

    for det in elements:

        print("Det{}".format(det)+os.linesep)



      










if __name__ == "__main__":
    main()