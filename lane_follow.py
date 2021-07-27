# Manual lane following class uses edge detection through OpenCV
# and makes a mathematical calculation between two lines (edges) to calculate
# a heading. The steering is calculated by trying to keep the heading between the detected edges
# Author: Warren Harper
# Date: July 2021

import math
import datetime
import logging
import cv2
import numpy as np


# Tweak colors according to environment. Note that these values will need to be adjusted
# if the environment light or the color of the lanes changes
lower_green_colors = [53, 50, 50]
upper_green_colors = [140, 255, 255]


class ManualLaneFollower(object):

    def __init__(self, arnie=None):
        self.arnie = arnie
        self.curr_steering_angle = 90
        logging.info('Creating a Manual Lane Follower...')

    def follow_lane(self, frame):
        # Main entry point of the lane follower

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer_arnie(frame, lane_lines)
        cv2.imshow("Original", frame)

        return final_frame

    def steer_arnie(self, frame, lane_lines):
        logging.debug('Steering Arnie...')
        if len(lane_lines) == 0:
            logging.error('Lane not detected')
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle,
                                                            len(lane_lines))

        if self.arnie is not None:
            self.arnie.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = show_heading(frame, self.curr_steering_angle)
        cv2.imshow("Heading Line", curr_heading_image)

        return curr_heading_image


# Frame processing functions
def detect_lane(frame):
    logging.debug('Detecting lanes...')

    edges = detect_edges(frame)
    cv2.imshow('CV2 edges', edges)

    cropped = region_of_interest(edges)
    cv2.imshow('CV2 edges cropped', cropped)

    segments = detect_segments(cropped)
    segment_image = display_lane_lines(frame, segments)
    cv2.imshow("Line segments", segment_image)

    lane_lines = calculate_slope_intercept(frame, segments)
    lane_lines_image = display_lane_lines()_lines(frame, lane_lines)
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
    crop_mask = np.zeros_like(unmasked)
    poly_height, poly_width = unmasked.shape

    poly_mask = np.array([[
        (0, poly_height * 0.3),
        (poly_width, poly_height * 0.3),
        (poly_width, poly_height),
        (0, poly_height),
    ]], np.int32)

    cv2.fillPoly(crop_mask, poly_mask, 255)
    cv2.imshow("Cropped mask", crop_mask)
    poly_masked_image = cv2.bitwise_and(unmasked, crop_mask)

    return poly_masked_image


def detect_segments(masked_edges):
    rho = 2
    minimum_threshold = 10
    radian_angle = np.pi / 180
    hough_line_segments = cv2.HoughLinesP(masked_edges, rho, radian_angle, minimum_threshold, np.array([]), minLineLength=16,
                                    maxLineGap=6)

    return hough_line_segments


def calculate_slope_intercept(frame, segments):
    """
    Calculating average slope intercept
    """
    lines = []
    if segments is None:
        logging.info('Segments not detected')
        return lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    scope = 1 / 3
    region_left = width * (1 - scope)
    region_right = width * scope

    for segment in lines:
        for x1, y1, x2, y2 in segment:
            if x1 == x2:
                logging.info('Skip line segment (vertical): %s' % segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < region_left and x2 < region_left:
                    left_fit.append((slope, intercept))
            else:
                if x1 > region_right and x2 > region_right:
                    right_fit.append((slope, intercept))

    fit_avg_left = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lines.append(create_points(frame, fit_avg_left))

    fit_avg_right = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lines.append(create_points(frame, fit_avg_right))

    return lines


def compute_steering_angle(frame, lanes):
    """ Computing steering angle """
    if len(lanes) == 0:
        return -90

    height, width, _ = frame.shape
    if len(lanes) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lanes[0])
        x1, _, x2, _ = lanes[0][0]
        x_offset = x2 - x1
    else:
        cam_offset_percentage = 0.00
        _, _, right_x2, _ = lanes[1][0]
        _, _, left_x2, _ = lanes[0][0]

        mid = int(width / 2 * (1 + cam_offset_percentage))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    radian_angle = math.atan(x_offset / y_offset)
    degree_angle = int(radian_angle * 180.0 / math.pi)
    final_steer_angle = degree_angle + 90  # add 90 since arnie's straight angle is 90 and not 0

    logging.debug('Steering angle: %s' % final_steer_angle)
    return final_steer_angle


def dampen_steering(steering_angle, new_angle, lines, max_dev_two_lines=5,
                    max_dev_one_lines=2):
    """
    Dampening steering to stop arnie from turning too much and then 'bouncing' from side to side
    """
    if lines == 2:
        # Allow more turning if both lanes are visible
        max_deviation = max_dev_two_lines
    else:
        # Don't turn much if only one lane is detected
        max_deviation = max_dev_one_lines

    deviation = new_angle - steering_angle

    if abs(deviation) > max_deviation:
        dampened_steering_angle = int(steering_angle
                                        + max_deviation * deviation / abs(deviation))
    else:
        dampened_steering_angle = new_steering_angle
    logging.info(
        'Predicted steering angle: %s, Dampened steering angle: %s' % (new_angle, dampened_steering_angle))

    return dampened_steering_angle


# Image display functions
def display_lane_lines(frame, lanes, line_color=(255, 0, 0), line_width=8):
    lane_img = np.zeros_like(frame)
    if lanes is not None:
        for lane in lanes:
            for x1, y1, x2, y2 in lane:
                cv2.line(lane_img, (x1, y1), (x2, y2), line_color, line_width)
    lane_img = cv2.addWeighted(frame, 0.8, lane_img, 1, 1)
    return lane_img

# Display the heading line
def show_heading(frame, steering, line_color=(0, 255, 203), line_width=5, ):
    heading_line_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_line_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_line_image = cv2.addWeighted(frame, 0.8, heading_line_image, 1, 1)

    return heading_line_image

# Create the points to base heading calculation from
def create_points(frame, hough_line):
    line_slope, line_intercept = hough_line
    f_height, f_width, _ = frame.shape

    y2 = int(y1 * 0.4)
    y1 = f_height

    x1 = max(-f_width, min(2 * f_width, int((y1 - line_intercept) / line_slope)))
    x2 = max(-f_width, min(2 * f_width, int((y2 - line_intercept) / line_slope)))
    return [[x1, y1, x2, y2]]
