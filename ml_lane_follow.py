# Program loads the trained machine learning model to process incoming images from Arnie
# the self-driving car, and makes a prediction on the steering angle
#
# Author: Warren Harper
# Date: July 2021

import logging
import math
import cv2
import numpy as np
from keras.models import load_model


class MlLaneFollower(object):

    def __init__(self, arnie=None, ):
        logging.info('Creating a machine learning lane follower')

        self.arnie = arnie
        self.curr_steering_angle = 90
        self.model = load_model('/home/pi/arnie/models/lane_follow/lane_navigation_final.h5')

    def follow_lane(self, frame):
        cv2.imshow("Original", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("Current steering angle = %d" % self.curr_steering_angle)

        if self.arnie is not None:
            self.arnie.front_wheels.turn(self.curr_steering_angle)
        main_frame = show_heading_line(frame, self.curr_steering_angle)

        return main_frame

    def compute_steering_angle(self, frame):
        """ Compute steering angle based on current frame """
        pre_processed_image = image_processor(frame)
        data = np.asarray([pre_processed_image])
        steering_angle = self.model.predict(data)[0]

        logging.debug('new steering angle: %s' % steering_angle)
        rounded_int = int(steering_angle + 0.5)

        return rounded_int


# Process images so they can be used by the ML model
def image_processor(img_unprocessed):
    height, _, _ = img_unprocessed.shape
    img_unprocessed = img_unprocessed[int(height / 2):, :, :]
    img_unprocessed = cv2.cvtColor(img_unprocessed, cv2.COLOR_BGR2YUV)
    img_unprocessed = cv2.GaussianBlur(img_unprocessed, (4, 4), 0)
    img_unprocessed = cv2.resize(img_unprocessed, (200, 66))
    img_processed = img_unprocessed / 254
    return img_processed


# Display the current heading line
def show_heading_line(frame, steering_angle, line_color=(0, 0, 254), line_thickness=6, ):
    height, width, _ = frame.shape
    arnie_heading = np.zeros_like(frame)

    steering_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_radian))
    y2 = int(height / 2)

    cv2.line(arnie_heading, (x1, y1), (x2, y2), line_color, line_thickness)
    arnie_heading = cv2.addWeighted(frame, 0.8, arnie_heading, 1, 1)

    return arnie_heading


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
