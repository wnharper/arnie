# Manual lane following class uses edge detection through OpenCV
# and makes a mathematical calculation between two lines (edges) to calculate
# a heading. The steering is calculated by trying to keep the heading between the detected edges
# Author: Warren Harper
# Date: July 2021

import math
import datetime
import sys
import cv2
import numpy as np
import logging

# Tweak colors according to environment. Note that these values will need to be adjusted
# if the environment light or the color of the lanes changes
lower_green_colors = [53, 50, 50]
upper_green_colors = [140, 255, 255]


class ManualLaneFollower(object):

    def __init__(self, car=None):
        logging.info('Creating a Manual Lane Follower...')
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        cv2.imshow("Original", frame)

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        logging.debug('Steering Arnie...')
        if len(lane_lines) == 0:
            logging.error('Lane not detected')
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle,
                                                            len(lane_lines))

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        cv2.imshow("Heading Line", curr_heading_image)

        return curr_heading_image


# Frame processing functions
def detect_lane(frame):
    logging.debug('Detecting lanes...')

    edges = detect_edges(frame)
    cv2.imshow('CV2 edges', edges)

    cropped_edges = region_of_interest(edges)
    cv2.imshow('CV2 edges cropped', cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    cv2.imshow("Line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    cv2.imshow("Lane lines", lane_lines_image)

    return lane_lines, lane_lines_image


# Convert and mask image
def detect_edges(frame):
    # filter for green lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Conversion", hsv)
    lower_green = np.array(lower_green_colors)
    upper_green = np.array(upper_green_colors)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow("Green mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


# crop to the lower half the screen
def region_of_interest(unmasked):
    height, width = unmasked.shape
    mask = np.zeros_like(unmasked)

    polygon = np.array([[
        (0, height * 0.3),
        (width, height * 0.3),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cv2.imshow("mask", mask)
    masked_image = cv2.bitwise_and(unmasked, mask)
    return masked_image


def detect_line_segments(cropped_edges):
    rho = 2
    angle = np.pi / 180
    min_threshold = 10
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=16,
                                    maxLineGap=6)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


def average_slope_intercept(frame, line_segments):
    """
    Calculating average slope intercept
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line should be on right 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)

    return lane_lines


def compute_steering_angle(frame, lane_lines):
    """ Computing steering angle """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.00
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90  # add 90 since picar's straight angle is 90 and not 0

    logging.debug('Steering angle: %s' % steering_angle)
    return steering_angle


def dampen_steering(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5,
                    max_angle_deviation_one_lane=2):
    """
    Dampening steering to stop car from turning too much and then 'bouncing' from side to side
    """
    if num_of_lane_lines == 2:
        # Allow more turning if both lanes are visible
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        # Don't turn much if only one lane is detected
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info(
        'Proposed steering angle: %s, Dampened steering angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle


# Utility Functions
def display_lines(frame, lines, line_color=(255, 0, 0), line_width=8):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 255, 203), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 0.4)

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]
