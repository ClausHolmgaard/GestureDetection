import sys
import cv2

#from helpers import *
from .helpers import *

# Text display positions
POSITIONS = {
    'info': (15, 20)
}


class SimpleDetect(object):
    def __init__(self):
        print(f"OpenCV version: {cv2.__version__    }")

        # Background frame
        self.bg = None

        # Kernel for erosion and dilation of masks
        self.kernel = np.ones((3,3),np.uint8)

        # Tracker
        self.tracker = None

        self.is_tracking = False

        # Capture device
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()

    def start(self):

        self.set_bg()

        while True:

            ret, frame = self.video.read()
            if not ret:
                continue

            diff = cv2.absdiff(self.bg, frame)
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Threshold the mask
            th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            # Opening, closing and dilation
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
            img_dilation = cv2.dilate(closing, self.kernel, iterations=2)
            # Get mask indexes
            imask = img_dilation > 0
            # Get foreground from mask
            foreground = mask_array(frame, imask)
            foreground_display = foreground.copy()

            bbox_initial = (60, 60, 170, 170)
            bbox = bbox_initial
            # Tracking status, -1 for not tracking, 0 for unsuccessful tracking, 1 for successful tracking
            tracking = -1

            # If tracking is active, update the tracker
            if self.is_tracking:
                tracking, bbox = self.tracker.update(foreground)
                tracking = int(tracking)

            hand_crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

            if self.is_tracking:
                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
                #cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

            tracking_string = f"Tracking result: {tracking}"
            cv2.putText(foreground_display, tracking_string, POSITIONS['info'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
            cv2.imshow("main", foreground_display)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                print("Exiting...")
                self.stop()
                break
            elif k == ord(' '):
                print("Updating background")
                self.set_bg()
            elif k == ord('t'):
                self.is_tracking = not(self.is_tracking)
                self.setup_tracker(2)
                tracking = self.tracker.init(frame, bbox)

    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()
        
    def set_bg(self):
        frame = get_newest_frame(self.video)
        self.bg = frame.copy()
    
    def setup_tracker(self, tracker_type):
        print("Starting tracker.")

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        tracker_type = tracker_types[tracker_type]

        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()

