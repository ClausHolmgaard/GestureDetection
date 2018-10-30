import tensorflow as tf

#from Hand3D.HandSegNet import HandSegNet
#from SimpleDetector.SimpleDetector import SimpleDetect
from YOLO import yolo

print(f"Tensorflow version: {tf.__version__}")


#h = HandSegNet()

#sd = SimpleDetect()
#sd.start()

y = yolo.YoloDetector()
