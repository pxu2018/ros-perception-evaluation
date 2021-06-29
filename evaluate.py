import io
import os

from file_utils import get_csv
from eval_utils import iou_2d, calc_distance

FILE_NAME = "/UwU.csv"
GT_FILE = "/UwU_gt.csv"


def main():

    elements = get_csv(os.getcwd()+ FILE_NAME)
    ground_tr = get_csv(os.getcwd()+ GT_FILE)
    

    # Iterations with gt instead of detections

    for gt in ground_tr:

        print("Gt:{}".format(gt))

        for el in elements:

            print(el)

            iou2d = iou_2d(gt.bbox2d,el.bbox2d)
            print(iou2d)

        #gt.asociate_detected_object(object=)










if __name__ == "__main__":
    main()