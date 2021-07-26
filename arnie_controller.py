# Manual keyboard controller for Arnie Self-driving car project
# Use WASD keys to control car.
# Images and corresponding steering angle will be saved to specified directory
# Author: Warren Harper
# Date: July 2021

import datetime
import logging
import sys
import termios
import time
import tty

import cv2
import picar


class ArnieController(object):

    def __init__(self):
        """ Initiating Arnie's camera and wheels"""

        # Initiate car
        picar.setup()

        # Set up camera
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 320)
        self.camera.set(4, 240)

        # Setup video recording
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        date_string = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.video_recorder('/home/pi/arnie/data/car_video%s.mp4' % date_string)

        # Set up back wheels
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0

        # Set up front wheels
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0
        self.front_wheels.turn(90)

    # video recording helper function
    def video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (320, 240))

    # Take keyboard input
    def getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def __enter__(self):
        """ Entering """
        return self

    def __exit__(self, _type, value, traceback):
        """ Exiting """
        if traceback is not None:
            print("Error occurred")

        self.cleanup()

    def cleanup(self):
        """ Resetting the hardware """
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_orig.release()
        cv2.destroyAllWindows()

    def drive(self, speed=0):
        """ Starting car according to speed arg (0-100) """

        self.back_wheels.speed = speed
        i = 0
        steering = 90
        speed = speed
        date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        while self.camera.isOpened():
            _, frame = self.camera.read()

            # Save image and corresponding steering angle
            self.video_orig.write(frame)
            cv2.imwrite("%s_%03d_%03d.png" % (date_time, i, steering), frame)

            i += 1

            char = self.getch()

            # Control car using WASD keys
            if char == "w":
                speed += 5
                if speed > 100:
                    speed = 100
                self.back_wheels.speed = speed

            if char == "s":
                speed -= 5
                if speed < 0:
                    speed = 0
                self.back_wheels.speed = speed

            if char == "d":
                steering += 5
                if steering > 135:
                    steering = 135
                self.front_wheels.turn(steering)

            if char == "a":
                steering -= 5
                if steering < 45:
                    steering = 45
                self.front_wheels.turn(steering)

            # Press q to quit program
            if char == "q":
                self.cleanup()
                break


def main():
    with ArnieController() as car:
        car.drive(0)


if __name__ == '__main__':
    main()
