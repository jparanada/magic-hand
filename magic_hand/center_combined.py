#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import cv2 as cv
import numpy as np

from camera_calibrate import read_offsets, embed_in_target_area, undistort, read_calibration, crop_with_offsets
from center import find_exact_corners, find_affine_matrix_for_centering, CONFIG_FINAL_W, CONFIG_FINAL_H
import skimage.transform as transform

PROPHOTO_GAMMA = 1.80078125


def center_combined(image_float, offsets_file, calibration_input_file, lower_hsv, upper_hsv):
    # Here we could just call camera_cal followed by center. However, that does two linear transforms back-to-back,
    # which slightly degrades image quality vs just doing one. If we're clever with our math we can find a single
    # perspective warp that is the composition of these two transforms (and also includes a rotation, if needed).
    # This is me being clever.
    (original_height, original_width, _) = np.shape(image_float)
    offsets = read_offsets(offsets_file)
    image_in_large_area = embed_in_target_area(image_float, offsets)
    calibration = read_calibration(calibration_input_file)
    # r_matrix and tvecs are only needed to calculate re-projection
    # error, so no need to grab those.
    camera_mtx = np.array(calibration["camera_matrix"])
    dist = np.array(calibration["dist"])
    perspective_matrix = np.array(calibration["perspective_matrix"])
    print("perspective_matrix opencv", perspective_matrix)
    perspective_src = np.array(calibration["perspective_src"])
    perspective_dst = np.array(calibration["perspective_dst"])
    tform = transform.ProjectiveTransform()
    tform.estimate(perspective_src, perspective_dst)
    perspective_matrix_skimage = tform.params

    undistorted_image_in_large_area, mapx, mapy = undistort(image_in_large_area, camera_mtx, dist)

    # we need this undistorted_image_in_large_area and then we would like to do a single transform to it,
    # followed by a crop to 4082x5652
    (large_height, large_width, _) = np.shape(undistorted_image_in_large_area)
    undistorted_correct_perspective_image_in_large_area = \
        cv.warpPerspective(undistorted_image_in_large_area, perspective_matrix,
                           (large_width, large_height), flags=cv.INTER_LINEAR)
    image_float_cropped = crop_with_offsets(undistorted_correct_perspective_image_in_large_area, original_width,
                                            original_height, offsets)

    image_8_cropped = np.uint8((image_float_cropped ** (1 / PROPHOTO_GAMMA)) * 255)
    exact_corners = find_exact_corners(image_8_cropped, lower_hsv, upper_hsv)
    [tl_cropped_before_affine, tr_cropped_before_affine, br_cropped_before_affine, bl_cropped_before_affine] = \
        exact_corners

    print("starting rotate_and_straighten...")
    # https://theailearner.com/tag/cv2-getperspectivetransform/
    affine_matrix = find_affine_matrix_for_centering(exact_corners)
    tl_final = np.matmul(affine_matrix, [tl_cropped_before_affine[0], tl_cropped_before_affine[1], 1])
    tr_final = np.matmul(affine_matrix, [tr_cropped_before_affine[0], tr_cropped_before_affine[1], 1])
    br_final = np.matmul(affine_matrix, [br_cropped_before_affine[0], br_cropped_before_affine[1], 1])
    bl_final = np.matmul(affine_matrix, [bl_cropped_before_affine[0], bl_cropped_before_affine[1], 1])

    target_x = offsets["target_x"]
    target_y = offsets["target_y"]
    roi_x = offsets["roi_x"]
    roi_y = offsets["roi_y"]

    x_offset = roi_x - target_x
    y_offset = roi_y - target_y
    roi_rotation = offsets["roi_rotation"]
    if roi_rotation == "cw":

        # the -1 is to compensate for the fact that the pixel at x_offset is itself the first pixel of (original) height
        origin_cropped_in_big_image = [x_offset+original_height-1, y_offset]
        tl_in_big_image_before_affine = [x_offset+original_height-1 - tl_cropped_before_affine[1],
                                         y_offset + tl_cropped_before_affine[0]]
        tr_in_big_image_before_affine = [x_offset+original_height-1 - tr_cropped_before_affine[1],
                                         y_offset + tr_cropped_before_affine[0]]
        br_in_big_image_before_affine = [x_offset+original_height-1 - br_cropped_before_affine[1],
                                         y_offset + br_cropped_before_affine[0]]
        # don't need this one, for debug only
        bl_in_big_image_before_affine = [x_offset+original_height-1 - bl_cropped_before_affine[1],
                                         y_offset + bl_cropped_before_affine[0]]

        corners_in_big_before_affine = [tl_in_big_image_before_affine, tr_in_big_image_before_affine,
                                        br_in_big_image_before_affine, bl_in_big_image_before_affine]

        # corners_in_big_before_affine_ints = np.array(corners_in_big_before_affine, dtype=np.int32)
        # print("corners_in_big_before_affine_ints", corners_in_big_before_affine_ints)
        # large_to_draw_on = np.uint8(255 * cv.pow(undistorted_correct_perspective_image_in_large_area, 1/PROPHOTO_GAMMA))
        # cv.imshow("large_to_draw_on", large_to_draw_on)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # for i in range(0, len(corners_in_big_before_affine_ints)):
        #     cv.line(large_to_draw_on, corners_in_big_before_affine_ints[i - 1], corners_in_big_before_affine_ints[i],
        #             (0, 0, 255), 5, cv.LINE_AA)
        # cv.imshow("corners_in_big_before_affine", large_to_draw_on)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    else:
        raise RuntimeError("I didn't write the logic for any rotation besides a cw one.")

    input_points = np.float32([tl_in_big_image_before_affine, tr_in_big_image_before_affine,
                               br_in_big_image_before_affine])
    output_points = np.float32([tl_final, tr_final, br_final])
    affine_matrix = cv.getAffineTransform(
        input_points,
        output_points
    )

    affine_matrix = np.append(affine_matrix, np.float32([[0, 0, 1]]), axis=0)
    combined_matrix = np.matmul(affine_matrix, perspective_matrix)
    return cv.warpPerspective(undistorted_image_in_large_area, combined_matrix, (CONFIG_FINAL_W, CONFIG_FINAL_H),
                              flags=cv.INTER_LINEAR)
