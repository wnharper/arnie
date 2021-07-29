# Main driver program for Arnie self-driving car
# Select feature by providing an argument(1,2,3 eg 'python arnie.py 2' will run ml mode:
# mode 1: manual_lane_follow uses the hand coded lane follower (non ML)
# mode 2: ml_lane_follow uses the trained machine learning model to follow lane
# mode 3: object_detect uses the machine learning model to detect objects
#
# Warning, the Raspberry Pi (even with the TPU) does not have the power to run all
# features at once unless the car is driving very slowly
#
# Author: Warren Harper
# Date: July 2021

from lane_follow import ManualLaneFollower
from ml_lane_follow import MlLaneFollower
from object_detection_processor import ObjectDetectionProcessor
import cv2
import picar
import logging
import datetime


class Arnie(object):

    def __init__(self, mode):
        """ Initiating Arnie's camera and wheels"""

        # take an argument of 1-3 in order to select the mode
        manual_lane_follow = False
        ml_lane_follow = False
        object_detect = False

        if mode == '1':
            manual_lane_follow = True
        elif mode == '2':
            ml_lane_follow = True
        elif mode == '3':
            object_detect = True
        else:
            print("not a valid argument, choose 1-3")
            quit()

        logging.info('Creating Arnie the self driving car...')

        date_string = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

        # Start/ setup the camera / video
        logging.debug(' Starting camera')
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 320)
        self.camera.set(4, 240)
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.video_main = self.video_recorder('/home/pi/DeepPiCar/driver/data/car_video%s.mp4' % date_string)

        # Car setup
        picar.setup()
        # Set back wheel speed to 0. Range is 0-100
        logging.debug('Setting back wheel speed to 0')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0

        # Set the front wheels straight (90) Range is 45 - 135
        logging.debug('Setting front wheels to straight (90)')
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0
        self.front_wheels.turn(90)

        # Initiate main processing units according to feature selection
        if manual_lane_follow:
            self.lane_follower = ManualLaneFollower(self)
            self.video_lane = self.video_recorder('/home/pi/arnie/data/car_video_lane%s.mp4' % date_string)

        if object_detect:
            self.object_detect_processor = ObjectsOnRoadProcessor(self)
            self.video_objects = self.video_recorder(
                '/home/pi/arnie/data/object_detection%s.mp4' % date_string)

        if ml_lane_follow:
            self.lane_follower = MlLaneFollower(self)
            self.video_lane = self.video_recorder('/home/pi/arnie/data/car_video_lane%s.mp4' % date_string)

        logging.info('Created Arnie')

    # Helper function to return a video recording object
    def video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (320, 240))

    def __enter__(self):
        """ Entering """
        return self

    def __exit__(self, _type, value, traceback):
        """ Exiting """
        if traceback is not None:
            # Exception occurred:
            logging.error('Exiting with exception %s' % traceback)

        self.cleanup_program()

    # Reset car, videos and windows when exiting program
    def cleanup_program(self):
        """ Resetting Arnie's hardware and closing windows """
        logging.info('Stopping arnie, resetting all hardware.')
        
        # Reset car
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        
        # close videos
        self.video_main.release()
        
        if __MANUAL_LANE_FOLLOW or __ML_LANE_FOLLOW:
            self.video_lane.release()

        if __OBJECT_DETECT:
            self.video_objects.release()
            
        # close all open windows    
        cv2.destroyAllWindows()

    def drive(self, speed):
        """ Driving car according to speed argument """

        logging.info('Current speed is %s...' % speed)
        self.back_wheels.speed = speed
        i = 0
        while self.camera.isOpened():
            _, main_image = self.camera.read()
            object_detect_image = main_image.copy()
            i += 1
            self.video_main.write(main_image)

            if __OBJECT_DETECT:
                object_detect_image = self.detect_objects_in_lane(object_detect_image)
                self.video_objects.write(object_detect_image)
                cv2.imshow('Detected Objects', object_detect_image)

            if __MANUAL_LANE_FOLLOW or __ML_LANE_FOLLOW:
                main_image = self.follow_lane(main_image)
                self.video_lane.write(main_image)
                cv2.imshow('Lanes / Path', main_image)
            
            # Quit program if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break

    def detect_objects_in_lane(self, image):
        image = self.traffic_sign_processor.detect_objects_in_lane(image)
        return image

    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image


def main(mode):
    if mode != 1 or not 2 or not 3:
        print("Not a valid mode")
        quit()

    with Arnie(mode) as car:
        car.drive(20)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')

    main(sys.argv[1])
