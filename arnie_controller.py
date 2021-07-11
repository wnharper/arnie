import logging
import picar
import cv2
import datetime
import sys, tty, termios, time

_SHOW_IMAGE = True


class ArnieController(object):
    __INITIAL_SPEED = 0
    __SCREEN_WIDTH = 320
    __SCREEN_HEIGHT = 240

    def __init__(self):
        """ Inititiating camera and wheels"""

        # Initiate car
        picar.setup()

        # Set up camera
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        # Set up back wheels
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        # Set up front wheels
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0  # calibrate servo to center
        self.front_wheels.turn(90)  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.create_video_recorder('/home/pi/DeepPiCar/driver/data/car_video%s.mp4' % datestr)

    def create_video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT))

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

    def drive(self, speed=__INITIAL_SPEED):
        """ Starting car according to speed arg (0-100) """

        self.back_wheels.speed = speed
        i = 0
        steering = 90
        speed = speed
        date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        while self.camera.isOpened():
            _, frame = self.camera.read()

            self.video_orig.write(frame)
            cv2.imwrite("%s_%03d_%03d.png" % (date_time, i, steering), frame)
            show_image('Arnie Vision', frame)

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


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def main():
    with ArnieController() as car:
        car.drive(20)


if __name__ == '__main__':
    main()
