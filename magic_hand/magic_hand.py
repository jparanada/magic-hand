#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import glob
import os

import argparse
import cv2 as cv
import numpy as np
from fix_color import scanner_refl_fix, cctiff
from white_balance import white_balance
from camera_calibrate import read_offsets, embed_in_target_area, undistort, read_calibration, crop_with_offsets
from center import find_exact_corners, find_affine_matrix_for_centering, CONFIG_FINAL_W, CONFIG_FINAL_H
from shrink_and_clip import shrink_and_clip
from srgb_and_corners import srgb_and_corners_pipeline
import skimage.transform as transform

CONFIG_SCANNER_REFL_FIX_CALIBRATION_PATH = "/Users/paranada/icc_profiles/scanner/scanner_cal.txt"
CONFIG_IMAGE_DPI = 1600
# defined by libtiff
COMPRESSION_NONE = 1

PROPHOTO_GAMMA = 1.80078125
SCALAR_16_BIT = 65535
SCALAR_8_BIT = 255


def run_pipeline(path: str, input_profile, source_wp_xyz, calibration_input_file, offsets_file, output_folder) -> None:
    """
    fix_color
    white_balance
    camera_calibrate
    center
    shrink_and_clip
    srgb_and_corners
    """
    filename = os.path.basename(path)
    full_save_path_no_extension = os.path.join(output_folder, filename).split(".tif")[0]

    path_f = scanner_refl_fix(path)
    image_f_pp = cctiff(input_profile, path_f, output_folder)
    # image_f_pp = cctiff(input_profile, "/Users/paranada/Pictures/pokemon-tcg/clean/hl-clean/hl018-gorebyss-clean_f.tif", output_folder)

    # filename = os.path.basename(image_f_pp)
    # full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    image = cv.imread(image_f_pp, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image[0, 0, 0], np.uint16)
    scalar = 65535 if is_16bit else 255
    image_float = image / scalar
    image_float **= PROPHOTO_GAMMA

    # convert BGR->RGB
    image_float = image_float[..., ::-1]

    print("starting white balance...")
    image_float = white_balance(image_float, source_wp_xyz)
    print("finished white balance.")

    # convert RGB->BGR
    image_float = image_float[..., ::-1]

    # Here we could just call camera_cal followed by center. However, that does two linear transforms back-to-back,
    # which slightly degrades image quality vs just doing one. If we're clever with our math we can find a single
    # perspective warp that is the composition of these two transforms (and also includes a rotation, if needed).
    # This is me being clever.
    # TODO: move this into some other file.
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
    undistorted_correct_perspective_image_in_large_area = cv.warpPerspective(undistorted_image_in_large_area, perspective_matrix,
                                                         (large_width, large_height), flags=cv.INTER_LINEAR)
    image_float_cropped = crop_with_offsets(undistorted_correct_perspective_image_in_large_area, original_width, original_height, offsets)

    image_8_cropped = np.uint8((image_float_cropped ** (1 / PROPHOTO_GAMMA)) * 255)
    exact_corners = find_exact_corners(image_8_cropped)
    [tl_cropped_before_affine, tr_cropped_before_affine, br_cropped_before_affine, bl_cropped_before_affine] = exact_corners

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
        tl_in_big_image_before_affine = [x_offset+original_height-1 - tl_cropped_before_affine[1], y_offset + tl_cropped_before_affine[0]]
        tr_in_big_image_before_affine = [x_offset+original_height-1 - tr_cropped_before_affine[1], y_offset + tr_cropped_before_affine[0]]
        br_in_big_image_before_affine = [x_offset+original_height-1 - br_cropped_before_affine[1], y_offset + br_cropped_before_affine[0]]
        # don't need this one, for debug only
        bl_in_big_image_before_affine = [x_offset+original_height-1 - bl_cropped_before_affine[1], y_offset + bl_cropped_before_affine[0]]

        corners_in_big_before_affine = [tl_in_big_image_before_affine, tr_in_big_image_before_affine, br_in_big_image_before_affine, bl_in_big_image_before_affine]

        # corners_in_big_before_affine_ints = np.array(corners_in_big_before_affine, dtype=np.int32)
        # print("corners_in_big_before_affine_ints", corners_in_big_before_affine_ints)
        # large_to_draw_on = np.uint8(255 * cv.pow(undistorted_correct_perspective_image_in_large_area, 1/PROPHOTO_GAMMA))
        # cv.imshow("large_to_draw_on",
        #           large_to_draw_on)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # for i in range(0, len(corners_in_big_before_affine_ints)):
        #     cv.line(large_to_draw_on, corners_in_big_before_affine_ints[i - 1], corners_in_big_before_affine_ints[i], (0, 0, 255), 5, cv.LINE_AA)
        # cv.imshow("corners_in_big_before_affine", large_to_draw_on)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    else:
        raise RuntimeError("I didn't write the logic for any rotation besides a cw one.")

    input_points = np.float32([tl_in_big_image_before_affine, tr_in_big_image_before_affine, br_in_big_image_before_affine])
    print("input", input_points)
    output_points = np.float32([tl_final, tr_final, br_final])
    print("output", output_points)
    affine_matrix = cv.getAffineTransform(
        input_points,
        output_points
    )

    affine_matrix = np.append(affine_matrix, np.float32([[0, 0, 1]]), axis=0)
    combined_matrix = np.matmul(affine_matrix, perspective_matrix)
    print("combined_matrix", combined_matrix)
    image_float = cv.warpPerspective(undistorted_image_in_large_area, combined_matrix, (CONFIG_FINAL_W, CONFIG_FINAL_H), flags=cv.INTER_LINEAR)

    print("starting shrink_and_clip...")
    image_float = shrink_and_clip(image_float)
    print("shrink_and_clip complete.")

    image_float **= 1 / PROPHOTO_GAMMA
    image_float *= scalar
    dst = np.uint16(image_float) if is_16bit else np.uint8(image_float)
    full_output_path = full_save_path_no_extension + "-mh_final.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, 295, cv.IMWRITE_TIFF_YDPI, 295, cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])

    srgb_and_corners_pipeline(full_output_path, output_folder)

    os.remove(full_output_path)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(description="Your favorite Rocket Gang Secret Mecha.")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    parser.add_argument(
        "-i", dest="calibration_input_file", type=str,
        help="input file that describes a calibration. for use only with -c",
        required=True)
    parser.add_argument(
        "-o", dest="output_folder", type=str, default=".",
        help="output folder for color-corrected images")
    parser.add_argument(
        "-f", dest="offsets_file", type=str,
        help="file that describes offsets of the distortion target and image ROIs. for use only with -c")
    parser.add_argument(
        "-p",
        dest="input_profile",
        type=str,
        help="input icc profile",
        required=True)
    parser.add_argument(
        "-s",
        dest="src_wp",
        type=str,
        help="source white point as X,Y,Z (scaled to 1)",
        required=True)
    args = parser.parse_args()
    source_wp_xyz = np.array(args.src_wp.split(","), dtype=np.float_)
    if len(source_wp_xyz) != 3:
        raise ValueError("source white point must be given as X,Y,Z")
    paths = []
    for i in args.path:
        image_path = glob.glob(i)
        if isinstance(image_path, list) and image_path:
            paths.extend(image_path)
        else:
            print(f"{i} is not a valid path, skipping")
    for image_path in paths:
        run_pipeline(image_path, args.input_profile, source_wp_xyz, args.calibration_input_file, args.offsets_file,
                     args.output_folder)
