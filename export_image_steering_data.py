# Program uses manual lane follower to detect edges in a recorded
# video and then saves it out as an image sequence along with current
# steering angle for each image
#
# Author: Warren Harper
# Date: July 2021

import cv2
import sys
from lane_follow import ManualLaneFollower


def save_steering_angle(video_file):
    # Create lane follow object
    lane_reader = ManualLaneFollower()
    capture = cv2.VideoCapture(video_file + '.mp4')

    try:
        i = 0
        while capture.isOpened():
            _, frame = capture.read()
            lane_reader.follow_lane(frame)

            # Save current image along with current steering angle
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_reader.curr_steering_angle), frame)
            i += 1

            # quit program if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    save_steering_angle(sys.argv[1])