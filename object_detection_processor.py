# Object detection program for Arnie self driving car
# Loads the trained model and initializes the Coral TPU for usage
# Uses CV2 to draw boxes around detected objects
# Author: Warren Harper
# Date: July 2021

import time
import edgetpu.detection.engine
import cv2
import logging
import datetime
from PIL import Image
from traffic_objects import *


class ObjectDetectionProcessor(object):
    """ Class detects objects in front of the car """

    def __init__(self,
                 car=None,
                 speed_limit=20,
                 model='/home/pi/arnie/models/object_detection/road_signs_quantized_1_edgetpu.tflite',
                 label='/home/pi/arnie/models/object_detection/road_sign_labels.txt',
                 width=640,
                 height=480):

        logging.info('Creating a ObjectsOnRoadProcessor...')
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

        # initialize edge TPU
        logging.info('Start Edge TPU with model %s...' % model)
        self.engine = edgetpu.detection.engine.DetectionEngine(model)
        self.min_confidence = 0.30
        self.num_of_objects = 3
        logging.info('Edge TPU initialization complete')

        # Start OpenCV for rectangle boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottom_left_corner = (10, height - 10)
        self.font_size = 1
        self.font_color = (255, 255, 255)
        self.line_width = 1
        self.line_type = 2
        self.box_color = (0, 0, 255)
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_display = 1.0  # ms

        # create detectable objects
        self.detectable_objects = {0: GreenLight(),
                                   1: RedLight(),
                                   2: Speed(55),
                                   3: Speed(15),
                                   4: Person()
                                   }

    def detect_object_main(self, frame):

        logging.debug('Detecting objects...')
        objects, final_frame = self.detect_objects(frame)
        self.control_arnie(objects)

        return final_frame

    def control_arnie(self, objects):
        logging.debug('Controlling car...')
        arnie_state = {"speed": self.speed_limit, "speed_limit": self.speed_limit}

        if len(objects) == 0:
            logging.debug('No objects detected, drive at speed limit of %s.' % self.speed_limit)

        for sign in objects:

            processor = self.traffic_objects[sign.label_id]
            if processor.is_close_by(sign, self.height):
                processor.set_car_state(arnie_state)

        self.resume_driving(arnie_state)

    # Steps for frame processing
    def detect_objects(self, frame):
        logging.debug('Detecting objects...')

        # query TPU
        start_ms = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_frame)
        objects = self.engine.DetectWithImage(image_pil, threshold=self.min_confidence, keep_aspect_ratio=True,
                                              relative_coord=False, top_k=self.num_of_objects)
        if objects:
            for signs in objects:
                box_height = signs.bounding_box[1][1] - signs.bounding_box[0][1]
                box_width = signs.bounding_box[1][0] - signs.bounding_box[0][0]
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" % (self.labels[signs.label_id], signs.score * 100, box_width, box_height))
                box = signs.bounding_box
                coordinate_top_left = (int(box[0][0]), int(box[0][1]))
                coordinate_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(frame, coordinate_top_left, coordinate_bottom_right, self.boxColor, self.boxLineWidth)
                text = "%s %.0f%%" % (self.labels[signs.label_id], signs.score * 100)
                coordinate_top_left = (coordinate_top_left[0], coordinate_top_left[1] + 15)
                cv2.putText(frame, text, coordinate_top_left, self.font, self.fontScale, self.boxColor,
                            self.lineType)
        else:
            logging.debug('Objects not detected')

        elapsed_time_milliseconds = time.time() - start_ms

        annotation = "%.1f FPS" % (1.0 / elapsed_time_milliseconds)
        logging.debug(annotation)
        cv2.putText(frame, annotation, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor,
                    self.lineType)
        cv2.imshow('Detected Objects', frame)

        return objects, frame

    def speed_setter(self, speed):
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed

    def resume_driving(self, arnie_state):
        old_speed = self.speed
        self.speed_limit = arnie_state['speed_limit']
        self.speed = arnie_state['speed']

        if self.speed == 0:
            self.speed_setter(0)
        else:
            self.speed_setter(self.speed_limit)
        logging.debug('Current Speed = %d, New Speed = %d' % (old_speed, self.speed))

        if self.speed == 0:
            # stop car for 2 seconds
            time.sleep(2)
