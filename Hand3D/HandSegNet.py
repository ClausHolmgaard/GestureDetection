"""
Segment a hand out from an image
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

class HandSegNet(object):
    def __init__(self):
        pass
    
    def MyModel(self):
        model = Sequential()

        

        return model
    
    def Hand3DModel(self):
        model = Sequential()

        # Block 1
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(255, 255, 3), padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        # Block 2
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        # Block 3
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        # Block 4
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))

        # Encoding
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, kernel_size=(1, 1), activation='relu', padding='same'))

        model.add(UpSampling2DBilinear(256, 256))

        return model