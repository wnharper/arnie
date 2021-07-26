# Program determines the actions taken when an object is detected
#
# Author: Warren Harper
# Date: July 2021

import logging
from threading import Timer


class DetectableObject(object):

    def set_arnie_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        obj_height = obj.bounding_box[1][1] - obj.bounding_box[0][1]
        return obj_height / frame_height > min_height_pct


class RedLight(DetectableObject):

    def set_arnie_state(self, car_state):
        logging.debug('Red detected, stopping')
        car_state['speed'] = 0


class GreenLight(DetectableObject):

    def set_arnie_state(self, car_state):
        logging.debug('Green detected, continue')


class Person(DetectableObject):

    def set_arnie_state(self, car_state):
        logging.debug('Person detected, stopping')
        car_state['speed'] = 0


class Speed(DetectableObject):

    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_arnie_state(self, car_state):
        car_state['speed_limit'] = self.speed_limit
        logging.debug('Set speed to %d' % self.speed_limit)
