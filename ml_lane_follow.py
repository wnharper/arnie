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

    def __init__(self, car=None,):
        logging.info('Creating a machine learning lane follower')

        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model('/home/pi/arnie/models/lane_follow/lane_navigation_final.h5')

    def follow_lane(self, frame):
        cv2.imshow("Original", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("Current steering angle = %d" % self.curr_steering_angle)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        main_frame = display_heading_line(frame, self.curr_steering_angle)

        return main_frame

    def compute_steering_angle(self, frame):
        """ Compute steering angle based on current frame """
        pre_processed_image = img_preprocess(frame)
        data = np.asarray([pre_processed_image])
        steering_angle = self.model.predict(data)[0]

        logging.debug('new steering angle: %s' % steering_angle)
        rounded_int = int(steering_angle + 0.5)
        return rounded_int


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_radian))
    y2 = int(height / 2)

    cv2.line(heading, (x1, y1), (x2, y2), line_color, line_width)
    heading = cv2.addWeighted(frame, 0.8, heading, 1, 1)

    return heading


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
