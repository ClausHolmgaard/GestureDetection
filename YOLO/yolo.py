#YOLO_PATH = r"/home/claus/git/keras-yolo3"

import cv2
import numpy as np
from PIL import Image
import time
import sys
#if YOLO_PATH not in sys.path:
#    sys.path.append(YOLO_PATH)

#from yolo import YOLO, detect_video
from KerasYolo3.yolo import YOLO, detect_video


class YoloDetector(object):
    def __init__(self):

        self.res_image = None
        #yolo_settings = {"model_path": 'model_data/yolo.h5',
        #                 "anchors_path": 'model_data/yolo_anchors.txt'}
        #self.yolo = YOLO(model_path = 'model_data/yolo3_tiny.h5',
        #                 anchors_path = 'model_data/tiny_yolo_anchors.txt')

        #self.yolo = YOLO(model_path = 'model_data/yolo3_tiny.h5',
        #                 anchors_path = 'model_data/tiny_yolo_anchors.txt')

        self.yolo = YOLO(model_path = 'output/trained_weights_final.h5',
                         anchors_path = 'model_data/tiny_yolo_anchors.txt',
                         classes_path = 'model_data/hand_classes',
                         model_image_size = (320, 320),
                         gpu_num = 0,
                         score = 0.3,
                         iou = 0.45)

        #self.yolo = YOLO()

        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()

        self.start()

    def start(self):
        while True:

            ret, frame = self.video.read()
            self.res_image = self.do_yolo(frame)

            cv2.imshow("Main", frame)
            if self.res_image is not None:
                cv2.imshow("Result", self.res_image)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                print("Exiting...")
                self.stop()
                break
            #if k == ord(' '):
            #    self.res_image = self.do_yolo(frame)
    
    def stop(self):
        pass

    def do_yolo(self, image):
        #print(image.shape)
        before = time.time()
        r_image = self.yolo.detect_image(Image.fromarray(image))
        after = time.time()

        #print(f"Processing time: {after - before}")
        return np.asarray(r_image)