# Object detection program for Arnie self driving car
# Loads the trained model and initializes the Coral TPU for usage
# Uses CV2 to draw boxes around detected objects
# Author: Warren Harper
# Date: July 2021

import edgetpu.detection.engine
import cv2
from PIL import Image
import logging
import time
import datetime
from detectable_objects import *


class ObjectDetectionProcessor(object):
    """ Detects objects in front of the car """

    def __init__(self,
                 arnie=None,
                 speed=20,
                 model='/home/pi/arnie/models/object_detection/road_signs_quantized_1_edgetpu.tflite',
                 label='/home/pi/arnie/models/object_detection/road_sign_labels.txt',
                 frame_width=320,
                 frame_height=240):

        logging.info('Creating an ObjectsOnRoadProcessor...')
        self.width = frame_width
        self.height = frame_height

        # Initialize arnie the self-driving car
        self.speed = speed
        self.speed_limit = speed
        self.arnie = arnie

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

        # initialize edge TPU
        logging.info('Start Edge TPU with model %s...' % model)
        self.engine = edgetpu.detection.engine.DetectionEngine(model)
        self.number_of_objects = 4
        self.minimum_confidence = 0.40
        logging.info('Edge TPU initialization complete')

        # Start OpenCV for rectangle boxes
        self.bottom_left_corner = (12, height - 12)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 1
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.font_color = (0, 0, 255)
        self.line_width = 1
        self.line_type = 2
        self.box_color = (0, 0, 255)
        self.time_to_display = 2.0

        # create detectable objects
        self.detectable_objects = {0: GreenLight(),
                                   1: RedLight(),
                                   2: Speed(55),
                                   3: Speed(15),
                                   4: Person()
                                   }

    def detect_object_main(self, frame):

        logging.debug('Detecting objects...')
        objects, final_detection_frame = self.detect(frame)
        self.control_arnie(objects)

        return final_detection_frame

    def control_arnie(self, objects):
        logging.debug('Controlling Arnie the self-driving car...')
        arnie_state = {"Arnie's speed = ": self.speed_limit, "Current speed limit = ": self.speed_limit}

        if len(objects) == 0:
            logging.debug('Nothing detected, proceed to drive at speed: %s.' % self.speed_limit)

        for sign in objects:

            processor = self.traffic_objects[sign.label_id]
            if processor.is_close_by(sign, self.height):
                processor.set_car_state(arnie_state)

        self.continue_driving(arnie_state)

    # Steps for frame processing
    def detect(self, frame):
        logging.debug('Detecting objects...')

        # query TPU
        start_ms = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_frame)
        detected_objects = self.engine.DetectWithImage(image_pil, threshold=self.minimum_confidence,
                                                       keep_aspect_ratio=True, relative_coord=False,
                                                       top_k=self.number_of_objects)
        if detected_objects:
            for signs in detected_objects:
                box_height = signs.bounding_box[1][1] - signs.bounding_box[0][1]
                box_width = signs.bounding_box[1][0] - signs.bounding_box[0][0]
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" % (
                self.labels[signs.label_id], signs.score * 100, box_width, box_height))
                box = signs.bounding_box
                coordinate_top_left = (int(box[0][0]), int(box[0][1]))
                coordinate_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(frame, coordinate_top_left, coordinate_bottom_right, self.boxColor, self.boxLineWidth)
                text = "%s %.0f%%" % (self.labels[signs.label_id], signs.score * 100)
                coordinate_top_left = (coordinate_top_left[0], coordinate_top_left[1] + 15)
                cv2.putText(frame, text, coordinate_top_left, self.font, self.fontScale, self.boxColor, self.lineType)
        else:
            logging.debug('Objects not detected')

        elapsed_time_milliseconds = time.time() - start_ms

        annotation = "%.1f FPS" % (1.0 / elapsed_time_milliseconds)
        logging.debug(annotation)
        cv2.putText(frame, annotation, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor,
                    self.lineType)
        cv2.imshow('Detected Objects', frame)

        return detected_objects, frame

    def speed_setter(self, speed):
        self.speed = speed
        if self.arnie is not None:
            self.arnie.back_wheels.speed = speed
            logging.debug("Setting Arnie's speed to: %d" % speed)

    def continue_driving(self, arnie_state):
        prev_speed = self.speed
        self.speed = arnie_state['speed']
        self.speed_limit = arnie_state['speed_limit']

        if self.speed == 0:
            self.speed_setter(0)
        else:
            self.speed_setter(self.speed_limit)

        logging.debug('Arnie\'s current speed: %d, Setting new speed to: %d' % (prev_speed, self.speed))

        if self.speed == 0:
            # stop car for 2 seconds
            time.sleep(2)
