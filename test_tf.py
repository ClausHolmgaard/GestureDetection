import tensorflow as tf

print("Is GPU available: " + str(tf.test.is_gpu_available()))

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())