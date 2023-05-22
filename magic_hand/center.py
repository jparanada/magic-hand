#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from collections import Counter
import glob
import math
import os

import cv2 as cv
import numpy as np

CONFIG_FINAL_W = 4082
CONFIG_FINAL_H = 5652
CONFIG_IMAGE_DPI = 1600
CONFIG_SCALE_FUDGE_FACTOR = 5699/5692


# defined by libtiff
COMPRESSION_NONE = 1

PROPHOTO_GAMMA = 1.80078125
# all HSV values are from border on ProPhoto g1.8 EX card "as-encoded" RGB values
# (231, 224, 105) -> np.array([56.67/2, 54.55/100*255, 90.59/100*255]) dark yellow, bottom
# HSV 57.3 61,2 87.2 EX border dark yellow again
# (220, 226, 160) -> np.array([57.39/2, 30.13/100*255, 89.8/100*255]) pale yellow
LOWER_HSV_EX_NONHOLO = np.array([51 / 2, 25 / 100 * 255, 80 / 100 * 255])
UPPER_HSV_EX_NONHOLO = np.array([61 / 2, 64 / 100 * 255, 100 / 100 * 255])

# for Wigglytuff ex HL
# 135 142 145 (bottom) -> 200.7 7.3 57.0
# 108 117 123 (right, left)
# 97 105 109 (right) -> 200.5 10.6 42.7
# 118 126 131 or 122 131 135 (top) -> 201.5 10.2 51.5 or 200.7 9.9 53.1
# 139 137 129 (inner bottom left) -> 48.7 6.7 54.4
# 135 138 143 (inner bottom by weakness) -> 218.8 5.8 56.2

# let's go with HSV 200 8.5 55, 193-207 5-12.5 40-64
LOWER_HSV_EX_EX = np.array([193 / 2, 5 / 100 * 255, 40 / 100 * 255])
UPPER_HSV_EX_EX = np.array([207 / 2, 12.5 / 100 * 255, 64 / 100 * 255])

# Wizards yellow: 176 163 39 -> [54.31 / 2, 77.84 / 100 * 255, 69.02 / 100 * 255]
LOWER_HSV_WIZARDS = np.array([49.31 / 2, 72.84 / 100 * 255, 64.02 / 100 * 255])
UPPER_HSV_WIZARDS = np.array([59.31 / 2, 82.84 / 100 * 255, 74.02 / 100 * 255])

LOWER_HSV = LOWER_HSV_EX_NONHOLO
UPPER_HSV = UPPER_HSV_EX_NONHOLO

TOP_BORDER_AREA = np.intp(1 / 3 * CONFIG_IMAGE_DPI)
# empirical value is 1.36; should be a bit higher than that. 1.43 is the normal ratio so should be much lower than that.
EX_SHORT_STRIP_HW_RATIO_THRESHOLD = 1.39

"""
1. find the 4 corners of the inner quadrilateral `exact_corners`
2. find the midpoints of this quadrilateral & get their intersection `center`
3. call cv.minAreaRect(exact_corners). rotate the image about center to straighten, using the theta returned by minAreaRect
4. translate the image so that center ends up at the exact midpoint of your final image
5. crop to final dimensions
"""


def load_image(img_path):
    image_16 = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    image_8 = cv.imread(img_path, cv.IMREAD_COLOR)

    return image_8, image_16


def get_theta(p1, p2):
    [p1_x, p1_y] = p1
    [p2_x, p2_y] = p2
    angle = math.degrees(math.atan2(p1_y - p2_y, p1_x - p2_x))
    return angle


def get_intersection_from_line_points(line1_p1, line1_p2, line2_p1, line2_p2):
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    [x1, y1] = np.double(line1_p1)
    [x2, y2] = np.double(line1_p2)
    [x3, y3] = np.double(line2_p1)
    [x4, y4] = np.double(line2_p2)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    # print("intersection", px, py)
    return np.array([px, py])


def get_intersection(horizontal_line, vertical_line):
    # https://stackoverflow.com/questions/383480/intersection-of-two-lines-defined-in-rho-theta-parameterization/383527#383527
    A = np.array([
        [-1, vertical_line[0] / vertical_line[1]],
        [-horizontal_line[1] / horizontal_line[0], 1]
    ])
    b = np.array([
        [vertical_line[0] / vertical_line[1] * vertical_line[3] - vertical_line[2]],
        [horizontal_line[3] - horizontal_line[1] / horizontal_line[0] * horizontal_line[2]]
    ])
    [x], [y] = np.linalg.solve(A, b)
    return [x, y]


def find_rough_corners_with_hough(image, thresh):
    # cv.imshow('Image thresh no morph', thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, None, iterations=4)
    # cv.imshow('Image thresh morph', thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    thresh = cv.bitwise_not(thresh)
    cv.imshow('Image thresh not', thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # cv.floodFill(thresh, None, (0, 0), 0)
    # cv.imshow('Image thresh flood', thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows

    # see https://docs.opencv.org/4.5.5/d4/d73/tutorial_py_contours_begin.html
    # https://docs.opencv.org/4.5.5/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # https://stackoverflow.com/questions/67457125/how-to-detect-white-region-in-an-image-with-opencv-python
    r = max(contours, key=cv.contourArea)

    tmp = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv.drawContours(tmp, [r], -1, 255)
    cv.imshow("after contour detection", tmp)
    cv.waitKey(0)
    cv.destroyAllWindows()

    linesP = cv.HoughLinesP(tmp, rho=1, theta=1 * np.pi / 180, threshold=7, minLineLength=200, maxLineGap=35)

    lines = [line_p[0] for line_p in linesP]
    vertical_lines = filter_on_theta_with_theta(lines, 90, threshold_degrees=0.35)
    horizontal_lines = filter_on_theta_with_theta(lines, 180, threshold_degrees=0.35)
    print("vertical lines", vertical_lines)
    print("horizontal lines", horizontal_lines)

    # cdstP = np.zeros(image.shape, np.uint8)  # cv.cvtColor(borders, cv.COLOR_GRAY2BGR)
    output = image.copy()
    top = [float("inf"), float("inf"), float("inf"), float("inf")]
    right = [-1, -1, -1, -1]
    bottom = [-1, -1, -1, -1]
    left = [float("inf"), float("inf"), float("inf"), float("inf")]
    # print("find_rough_corners_with_hough -- all lines:")
    # for l in lines:
    #     print(l)
    #     [x1, y1, x2, y2] = l
    #     cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv.LINE_AA)
    for l in vertical_lines:
        print(l)
        [x1, _, x2, _] = l
        if x1 < left[0] or x2 < left[2]:
            left = l
        if x1 > right[0] or x2 > right[2]:
            right = l
    for l in horizontal_lines:
        print(l)
        [_, y1, _, y2] = l
        if y1 < top[1] or y2 < top[3]:
            top = l
        if y1 > bottom[1] or y2 > bottom[3]:
            bottom = l

    for l in [top, right, bottom, left]:
        cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv.LINE_AA)
    print("four lines", [top, right, bottom, left])
    cv.imshow("selected lines (rough)", output)
    cv.waitKey(0)
    cv.destroyAllWindows()

    tl = get_intersection_from_line_points(
        [top[0], top[1]], [top[2], top[3]], [left[0], left[1]], [left[2], left[3]])
    tr = get_intersection_from_line_points(
        [top[0], top[1]], [top[2], top[3]], [right[0], right[1]], [right[2], right[3]])
    br = get_intersection_from_line_points(
        [bottom[0], bottom[1]], [bottom[2], bottom[3]], [right[0], right[1]], [right[2], right[3]])
    bl = get_intersection_from_line_points(
        [bottom[0], bottom[1]], [bottom[2], bottom[3]], [left[0], left[1]], [left[2], left[3]])
    corners = np.array([tl, tr, br, bl])
    print("corners", corners)
    for i in range(4):
        cv.line(output, np.int32(corners[i - 1]), np.int32(corners[i]), (255, 0, 0), 3, cv.LINE_AA)
    cv.imshow("fully-extended selected lines (rough)", output)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return corners


def blur(image_8, k=15):
    # can try no blur, with the intent that descreen is a low-pass filter already
    image_8_blurred = image_8
    # image_8_blurred = cv.GaussianBlur(image_8, (k, k), 0)
    # image_8_blurred = cv.fastNlMeansDenoisingColored(image_8)
    # image_8_blurred = cv.bilateralFilter(image_blurred, -1, 24, 16)

    # cv.imshow("blurred", image_blurred)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return image_8_blurred


# TODO: idea: if the top and bottom are solid-colored, could use a tighter threshold for those areas
def get_thresholded_border(image_8, lower_hsv, upper_hsv):
    image_8_blurred = blur(image_8)

    # https://docs.opencv.org/4.5.5/da/d97/tutorial_threshold_inRange.html
    # https://docs.opencv.org/4.7.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
    # 0 ≤ H ≤ 180, 0 ≤ S ≤ 255, 0 ≤ V ≤ 255
    # Note HSV math does not care about color profile, it just works on whatever pixel values you give it.
    # That's how online calculators like https://colorizer.org/ work as well.
    # Let's just pass a 1.8 gamma-encoded ProPhoto image to the conversion and set threshold values accordingly. 
    hsv_image = cv.cvtColor(image_8_blurred, cv.COLOR_BGR2HSV)
    # black on unprofiled EX card is np.array([60.77/2, 10.54/100*255, 13.27/100*255])
    # border on unprofiled EX card is np.array([50.64/2, 53.29/100*255, 77.68/100*255])
    # having the V window lower thresh be low so that the border is as close to the inner border as possible
    # produces best results

    thresholded_border = cv.inRange(hsv_image, lower_hsv, upper_hsv)

    # TODO: this works sometimes but other times not robust and fills too much, destroying border info.
    # consider replacing with just morphologyEx?
    # cv.floodFill(thresholded_border, None, (0, 0), 255)

    thresholded_border = cv.morphologyEx(thresholded_border, cv.MORPH_CLOSE, None, thresholded_border, iterations=4)

    # cv.imshow("binarized, thresholded from hsv", thresholded_border)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return thresholded_border


# https://stackoverflow.com/questions/51689127/python-opencv-perspective-correction-for-rectangle-with-rounded-corners
def find_border_sobel(thresholded_border, retr_mode=cv.RETR_CCOMP, approx_mode=cv.CHAIN_APPROX_SIMPLE):
    # Use a edge detector on the segmentation image in order to find the contours
    sx = cv.Sobel(thresholded_border, cv.CV_32F, 1, 0)
    sy = cv.Sobel(thresholded_border, cv.CV_32F, 0, 1)
    m = cv.magnitude(sx, sy)

    # cv.imshow("sobel", m)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Refine the contours thickness
    m = cv.normalize(m, m, 0., 255., cv.NORM_MINMAX, cv.CV_8U)

    # cv.imshow("normalize", m)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    m = cv.ximgproc.thinning(m, m)

    # cv.imshow("thinning", m)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    tmp = np.zeros(m.shape, np.uint8)
    # Not robust at all lmao. Blacking out the outer edge.
    crop_width = 3800
    crop_height = 5415
    tmp[138:138 + crop_height, 178:178 + crop_width] = m[138:138 + crop_height, 178:178 + crop_width]

    inner_black_width = 3498
    inner_black_height = 4620
    tmp[413:413 + inner_black_height, 320:320 + inner_black_width] = 0

    contours, hierarchy = cv.findContours(tmp, retr_mode, approx_mode)
    # r = max(contours, key=cv.contourArea)
    tmp = np.zeros(m.shape, np.uint8)
    borders = cv.drawContours(tmp, contours, -1, 255, hierarchy=hierarchy)
    # cv.imshow("findContours", borders)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return borders, contours


def find_lines_hough_lines_p(shape, borders):
    linesP = cv.HoughLinesP(borders, rho=1, theta=1 * np.pi / 180, threshold=7, minLineLength=200, maxLineGap=35)
    cdstP = np.zeros(shape, np.uint8)  # cv.cvtColor(borders, cv.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv.LINE_AA)
    cv.imshow("find_lines_hough_lines_p Detected Lines - Standard Hough Line Transform", cdstP)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return linesP, cdstP


def filter_on_theta_with_theta(lines, theta, threshold_degrees=1.5):
    return np.array(filter_on_theta(None, None, lines, threshold_degrees, force_theta=theta))


def filter_on_theta(target_p1, target_p2, lines, threshold_degrees=1.5, force_theta=None):
    filtered_lines = []
    theta_target = force_theta if force_theta is not None else get_theta(target_p1, target_p2)
    for line in lines:
        [p1_x, p1_y, p2_x, p2_y] = line
        theta = get_theta([p1_x, p1_y], [p2_x, p2_y])

        # mod 180, not 360, because we don't care about vector directionality
        theta_target_adjusted = (theta_target + 360) % 180
        theta_adjusted = (theta + 360) % 180

        if abs(theta_target_adjusted - theta_adjusted) < threshold_degrees:
            filtered_lines.append(line)
        elif theta_adjusted < 180:
            if abs(theta_target_adjusted - (theta_adjusted + 180)) < threshold_degrees:
                filtered_lines.append(line)
            elif theta_target_adjusted < 180:
                if abs(theta_target_adjusted + 180 - theta_adjusted) < threshold_degrees:
                    filtered_lines.append(line)
                else:
                    print(f"rejected {theta_adjusted} vs reference {theta_target_adjusted}")
        else:
            print(f"rejected {theta_adjusted} vs reference {theta_target_adjusted}")

    return filtered_lines


# TODO it's goofy to have these lines, then draw them, then contour-detect, then detect a line.
# This might result in loss of precision since we only draw on integer pixels and the contour detection only returns
# integer pixels.
# Is there something more elegant we can do?
def get_line_from_filtered_lines(lines_filtered, height, width):
    temp_image = np.zeros((height, width), np.uint8)
    for line in lines_filtered:
        cv.line(temp_image, (line[0], line[1]), (line[2], line[3]), 255, 1, cv.LINE_AA)
    # probably don't need the below three lines, and it slows down the code a fair bit.
    # the idea was to deal with contours that are thicker than others, but that might not actually be a problem.

    # temp_image = cv.morphologyEx(temp_image, cv.MORPH_CLOSE, None, iterations=4)
    # temp_image = cv.normalize(temp_image, None, 0., 255., cv.NORM_MINMAX, cv.CV_8U)
    # temp_image = cv.ximgproc.thinning(temp_image)

    # cv.imshow("temp_image", temp_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    contours, _ = cv.findContours(temp_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # print(f"contours {contours}")
    fit_line_input = np.vstack(contours)
    # print(f"line contours {fit_line_input}")
    # https://docs.opencv.org/4.6.0/dd/d49/tutorial_py_contour_features.html
    line = cv.fitLine(fit_line_input, cv.DIST_L2, 0, 0.01, 0.01)
    print(f"line {[item[0] for item in line]}")
    return [item[0] for item in line]


def boujee_line_detection(hough_lines, area_to_look, dims, width_around_area=5):
    [tl, tr, br, bl] = area_to_look

    width = dims[1]
    height = dims[0]
    print(f"dims {dims}")
    # dims (5692, 4134, 3)

    top_lines = []
    left_lines = []
    right_lines = []
    bottom_lines = []

    # given a line between p1 & p2, the shortest distance from p3 to that line is
    # d = np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)
    # hough_lines look like e.g. [ [[1 2 3 4]] [[5 6 7 8]] [[x1 y1 x2 y2]] ]
    left_norm = np.linalg.norm(tl - bl)
    right_norm = np.linalg.norm(tr - br)
    top_norm = np.linalg.norm(tl - tr)
    bottom_norm = np.linalg.norm(bl - br)

    for hough_line_list in hough_lines:
        hough_line = hough_line_list[0]
        [p1_x, p1_y, p2_x, p2_y] = hough_line
        # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        p1 = np.array([p1_x, p1_y])
        p2 = np.array([p2_x, p2_y])
        if np.abs(np.cross(tl - bl, p1 - bl)) / left_norm <= width_around_area and np.abs(
                np.cross(tl - bl, p2 - bl)) / left_norm <= width_around_area:
            left_lines.append(hough_line)
        if np.abs(np.cross(tr - br, p1 - br)) / right_norm <= width_around_area and np.abs(
                np.cross(tr - br, p2 - br)) / right_norm <= width_around_area:
            right_lines.append(hough_line)
        if np.abs(np.cross(tl - tr, p1 - tr)) / top_norm <= width_around_area and np.abs(
                np.cross(tl - tr, p2 - tr)) / top_norm <= width_around_area:
            top_lines.append(hough_line)
        if np.abs(np.cross(bl - br, p1 - br)) / bottom_norm <= width_around_area and np.abs(
                np.cross(bl - br, p2 - br)) / bottom_norm <= width_around_area:
            bottom_lines.append(hough_line)

    # TODO consider np.vectorize and then filtering after that, or some way to use np.atan2 on a whole np.array instead of looping
    # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    # x = np.array([1, 2, 3, 4, 5])
    # f = lambda x: x ** 2
    # squares = f(x)
    top_lines_filtered = filter_on_theta(tl, tr, top_lines, force_theta=0)
    top_line = get_line_from_filtered_lines(top_lines_filtered, height, width)

    left_lines_filtered = filter_on_theta(tl, bl, left_lines, force_theta=90)
    left_line = get_line_from_filtered_lines(left_lines_filtered, height, width)

    right_lines_filtered = filter_on_theta(tr, br, right_lines, force_theta=90)
    right_line = get_line_from_filtered_lines(right_lines_filtered, height, width)

    bottom_lines_filtered = filter_on_theta(bl, br, bottom_lines, force_theta=0)
    bottom_line = get_line_from_filtered_lines(bottom_lines_filtered, height, width)

    temp_image = np.zeros((height, width), np.uint8)
    all_lines_filtered = []
    for l in top_lines_filtered:
        all_lines_filtered.append(l)
    for l in left_lines_filtered:
        all_lines_filtered.append(l)
    for l in right_lines_filtered:
        all_lines_filtered.append(l)
    for l in bottom_lines_filtered:
        all_lines_filtered.append(l)
    for l in all_lines_filtered:
        cv.line(temp_image, (l[0], l[1]), (l[2], l[3]), 255, 1, cv.LINE_AA)
    # cv.imshow("all_lines_filtered before best fit", temp_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    tl_exact = get_intersection(top_line, left_line)
    tr_exact = get_intersection(top_line, right_line)
    br_exact = get_intersection(bottom_line, right_line)
    bl_exact = get_intersection(bottom_line, left_line)
    result = np.array([tl_exact, tr_exact, br_exact, bl_exact])

    # print(f"exact corners result {result}")
    # tmp = np.zeros((height, width), np.uint8)
    # for i in range(4):
    #     cv.line(tmp, np.intp(result[i - 1]), np.intp(result[i]), 255, 5, cv.LINE_AA)
    # cv.imshow("boujee_line_detection lines only", tmp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return result


def get_y_offset(rect_w, rect_h):
    if rect_h > rect_w:
        hw_ratio = rect_h / rect_w
    else:
        hw_ratio = rect_w / rect_h
    # 1.43 is "normal" h/w ratio
    # 1.36 is "short" e-reader strip h/w ratio
    # print("rotated rect w, h, ratio", rect_w, rect_h, hw_ratio)

    if hw_ratio < EX_SHORT_STRIP_HW_RATIO_THRESHOLD:
        return -124
    return 0


def find_affine_matrix_for_centering(exact_corners):
    # float32 required
    rect = cv.minAreaRect(np.array(exact_corners, np.float32))
    # print("minAreaRect", rect)

    line1_p1 = (exact_corners[0] + exact_corners[1]) / 2  # top
    line2_p1 = (exact_corners[1] + exact_corners[2]) / 2  # right
    line1_p2 = (exact_corners[2] + exact_corners[3]) / 2  # bottom
    line2_p2 = (exact_corners[3] + exact_corners[0]) / 2  # left
    [mid_x, mid_y] = get_intersection_from_line_points(line1_p1, line1_p2, line2_p1, line2_p2)

    # print("mid x and y", mid_x, mid_y)

    (rect_w, rect_h) = rect[1]
    y_offset = get_y_offset(rect_w, rect_h)

    # Because we are rotating about mid_x, mid_y, that point stays the same post-rotation.
    np.set_printoptions(suppress=True)
    affine_matrix = cv.getRotationMatrix2D(
        (mid_x, mid_y), rect[2] - 90 if rect[2] > 45 else rect[2], CONFIG_SCALE_FUDGE_FACTOR)
    # print("affine_matrix (rotation matrix)", affine_matrix)
    x_translation = (CONFIG_FINAL_W / 2) - mid_x
    y_translation = (CONFIG_FINAL_H / 2) - mid_y + y_offset
    # Adding the translations this way to the rotation matrix is equivalent to rotation first,
    # followed by translation.
    affine_matrix[0, 2] += x_translation
    affine_matrix[1, 2] += y_translation
    print("affine_matrix rotation-then-translation matrix", affine_matrix)
    return affine_matrix


def rotate_and_straighten(image_16, exact_corners):
    # The full magic_hand combines this warp with the perspective warp that precedes it, which is technically better
    # (one fewer linear interpolation). This is to test just the centering piece.
    affine_matrix = find_affine_matrix_for_centering(exact_corners)
    image_16 = cv.warpAffine(image_16, affine_matrix, (CONFIG_FINAL_W, CONFIG_FINAL_H), flags=cv.INTER_LINEAR)

    # cv.imshow("post-translation FINAL", image_16)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return image_16


def get_fit_line_ransac(points):
    # see https://stackoverflow.com/questions/11722569/opencv-line-fitting-algorithm/15184899#15184899
    # for more on distType choice. DIST_L1 is "some sort of RANSAC fit", which is what I want.
    line = cv.fitLine(np.array(points), cv.DIST_L1, 0, 0.01, 0.01)
    return [item[0] for item in line]


def find_inner_edges_ransac(thresholded_border, shape):
    # we need to densify the contour, so CHAIN_APPROX_NONE is correct.
    _, border_contours = find_border_sobel(thresholded_border, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # list of xy coordinates, where a coordinate is an np.array of type int32 and length 2
    coordinates = [point[0] for contour in border_contours for point in contour]

    w = shape[1]
    h = shape[0]
    # heuristic to find a good starting point for the 4 best-fits we want to do
    # probably terrible if the image is too rotated, but for ruler-aligned cards, it won't be an issue.
    x_coords, y_coords = zip(*coordinates)
    left_x_coords = []
    right_x_coords = []
    for x in x_coords:
        if x < w/2:
            left_x_coords.append(x)
        else:
            right_x_coords.append(x)
    top_y_coords = []
    bottom_y_coords = []
    for y in y_coords:
        if y < h/2:
            top_y_coords.append(y)
        else:
            bottom_y_coords.append(y)
    # Use Counter to get counts of x-coordinates and y-coordinates
    x_left = Counter(left_x_coords).most_common(8)
    x_right = Counter(right_x_coords).most_common(8)
    y_top = Counter(top_y_coords).most_common(8)
    y_bottom = Counter(bottom_y_coords).most_common(8)
    # print(f"lrtb {x_left} {x_right} {y_top} {y_bottom}")
    x_left = Counter(left_x_coords).most_common(1)[0][0]
    x_right = Counter(right_x_coords).most_common(1)[0][0]
    y_top = Counter(top_y_coords).most_common(1)[0][0]
    y_bottom = Counter(bottom_y_coords).most_common(1)[0][0]
    # print(f"lrtb {x_left} {x_right} {y_top} {y_bottom}")
    x_left_points = []
    x_right_points = []
    y_top_points = []
    y_bottom_points = []
    margin_for_fit_line = 10
    for coordinate in coordinates:
        (x, y) = coordinate
        if x_left-margin_for_fit_line < x < x_left+margin_for_fit_line:
            x_left_points.append(coordinate)
        elif x_right-margin_for_fit_line < x < x_right+margin_for_fit_line:
            x_right_points.append(coordinate)
        if y_top-margin_for_fit_line < y < y_top+margin_for_fit_line:
            y_top_points.append(coordinate)
        if y_bottom-margin_for_fit_line < y < y_bottom+margin_for_fit_line:
            y_bottom_points.append(coordinate)
    # left_line = cv2.fitLine(np.array(x_left_points), cv.DIST_HUBER, 0, 0.01, 0.01)
    # right_line = cv2.fitLine(np.array(x_right_points), cv.DIST_HUBER, 0, 0.01, 0.01)
    # top_line = cv2.fitLine(np.array(y_top_points), cv.DIST_HUBER, 0, 0.01, 0.01)
    # bottom_line = cv2.fitLine(np.array(y_bottom_points), cv.DIST_HUBER, 0, 0.01, 0.01)
    left_line = get_fit_line_ransac(x_left_points)
    right_line = get_fit_line_ransac(x_right_points)
    top_line = get_fit_line_ransac(y_top_points)
    bottom_line = get_fit_line_ransac(y_bottom_points)
    tl_exact = get_intersection(top_line, left_line)
    tr_exact = get_intersection(top_line, right_line)
    br_exact = get_intersection(bottom_line, right_line)
    bl_exact = get_intersection(bottom_line, left_line)
    result = np.array([tl_exact, tr_exact, br_exact, bl_exact])

    # print(f"exact corners result {result}")
    # tmp = np.zeros(shape, np.uint8)
    # for i in range(4):
    #     cv.line(tmp, np.intp(result[i - 1]), np.intp(result[i]), 255, 5, cv.LINE_AA)
    # cv.imshow("find_inner_edges_ransac", tmp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return result


def find_exact_corners_with_old_algorithm(thresholded_border, image_8, image_8_blurred):
    border_sobel, _ = find_border_sobel(thresholded_border)
    hough_lines, border_sobel_hough_lines = find_lines_hough_lines_p(image_8_blurred.shape, border_sobel)
    # corners = find_corners(border_sobel, bounding_rect)
    corners = find_rough_corners_with_hough(image_8, thresholded_border)
    # print("corners", corners)
    exact_corners = boujee_line_detection(hough_lines, corners, image_8.shape, 4)
    exact_corners_ints = np.array(exact_corners, dtype=np.int32)
    # print("exact_corners_ints", exact_corners_ints)
    for i in range(0, len(exact_corners_ints)):
        cv.line(image_8, exact_corners_ints[i - 1], exact_corners_ints[i], (0, 0, 255), 5, cv.LINE_AA)
    cv.imshow("boujee_line_detection lines on pic", image_8)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return exact_corners


def find_exact_corners(image_8):
    thresholded_border = get_thresholded_border(image_8, LOWER_HSV, UPPER_HSV)
    return find_inner_edges_ransac(thresholded_border, image_8.shape)


def center_pipeline(img_file, save_path):
    filename = os.path.basename(img_file)
    # full_save_path = os.path.join(save_path, filename)
    full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    image_8, image_16 = load_image(img_file)
    is_16bit = isinstance(image_16[0][0][0], np.uint16)
    scalar = 65535 if is_16bit else 255

    image_16 = cv.pow(image_16 / scalar, PROPHOTO_GAMMA)

    # bounding_rect = find_rough_corners(image_8)
    # thresholded_border = np.zeros((h, w), dtype=np.uint8)
    # ex_dark_yellow_low = np.array([51.67 / 2, 49.55 / 100 * 255, 85.59 / 100 * 255])
    # ex_dark_yellow_high = np.array([61.67 / 2, 59.55 / 100 * 255, 95.59 / 100 * 255])
    #     tmp[138:138+crop_height, 178:178+crop_width] = m[138:138+crop_height, 178:178+crop_width]
    # TODO: I'm pretty sure these top border thresholds are too aggressive; okay for Dragon but not TRR
    # top_zoom = image_8_blurred[0:TOP_BORDER_AREA, 0:w]
    # thresholded_border[0:TOP_BORDER_AREA, 0:w] = get_thresholded_border(
    #     top_zoom, ex_dark_yellow_low, ex_dark_yellow_high)
    # rest_of_card = image_8_blurred[TOP_BORDER_AREA - 100:h, 0:w]
    # thresholded_border[TOP_BORDER_AREA - 100:h, 0:w] = get_thresholded_border(
    #     rest_of_card, LOWER_HSV_EX, UPPER_HSV_EX)

    exact_corners = find_exact_corners(image_8)

    exact_corners_ints = np.array(exact_corners, dtype=np.int32)
    # print("exact_corners_ints", exact_corners_ints)
    for i in range(0, len(exact_corners_ints)):
        cv.line(image_8, exact_corners_ints[i - 1], exact_corners_ints[i], (0, 0, 255), 5, cv.LINE_AA)
    cv.imshow("find_inner_edges_ransac lines on pic", image_8)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # exact_corners = find_exact_corners_with_old_algorithm(thresholded_border, image_8, image_8_blurred)

    image_16 = rotate_and_straighten(image_16, exact_corners)

    if is_16bit:
        dst = np.uint16(scalar * cv.pow(image_16, 1 / PROPHOTO_GAMMA))
    else:
        dst = np.uint8(scalar * cv.pow(image_16, 1 / PROPHOTO_GAMMA))
    full_output_path = full_save_path_no_extension + "-centered.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, CONFIG_IMAGE_DPI, cv.IMWRITE_TIFF_YDPI, CONFIG_IMAGE_DPI,
                cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="find center of an undistorted card, and Euclidean transform accordingly.")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    parser.add_argument(
        "-o",
        dest="output_folder",
        type=str,
        help="output folder for centered images",
        required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.output_folder):
        raise ValueError("Output path is not a folder")
    paths = []
    for i in args.path:
        path = glob.glob(i)
        if isinstance(path, list) and path:
            paths.extend(path)
    if not paths:
        raise ValueError("no valid paths were provided")
    for i in paths:
        i = os.path.abspath(i)
        if os.path.exists(i):
            center_pipeline(i, args.output_folder)
        else:
            raise ValueError("{} does not exist".format(i))
