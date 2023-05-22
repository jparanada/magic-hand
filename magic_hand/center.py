#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from collections import Counter
import glob
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
Y_OFFSET_FOR_SHORT_STRIP = -124

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


def get_y_offset(rect_w, rect_h):
    if rect_h > rect_w:
        hw_ratio = rect_h / rect_w
    else:
        hw_ratio = rect_w / rect_h
    # 1.43 is "normal" h/w ratio
    # 1.36 is "short" e-reader strip h/w ratio
    # print("rotated rect w, h, ratio", rect_w, rect_h, hw_ratio)

    if hw_ratio < EX_SHORT_STRIP_HW_RATIO_THRESHOLD:
        return Y_OFFSET_FOR_SHORT_STRIP
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

    exact_corners = find_exact_corners(image_8)

    exact_corners_ints = np.array(exact_corners, dtype=np.int32)
    # print("exact_corners_ints", exact_corners_ints)
    for i in range(0, len(exact_corners_ints)):
        cv.line(image_8, exact_corners_ints[i - 1], exact_corners_ints[i], (0, 0, 255), 5, cv.LINE_AA)
    cv.imshow("find_inner_edges_ransac lines on pic", image_8)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
