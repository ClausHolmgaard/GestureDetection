import numpy as np
import cv2


def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def get_newest_frame(cap_device):
    """
    ONLY USE THIS WHEN GETTING SINGLE FRAMES
    """
    #Purge the buffer
    for i in range(10): #Annoyingly arbitrary constant
        cap_device.grab()

    # Read first frame
    ok, frame = cap_device.read()
    if not ok:
        print("Cannot read video")
    
    return frame

def cv_to_plot(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
