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
from center_combined import center_combined
from config import get_config
from fix_color import scanner_refl_fix, cctiff
from white_balance import white_balance
from shrink_and_clip import shrink_and_clip
from srgb_and_corners import srgb_and_corners_pipeline

CONFIG_IMAGE_DPI = 1600
# defined by libtiff
COMPRESSION_NONE = 1

PROPHOTO_GAMMA = 1.80078125
SCALAR_16_BIT = 65535
SCALAR_8_BIT = 255


def run_pipeline(path: str, input_profile, config, calibration_input_file, offsets_file, output_folder,
                 is_quick) -> None:
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

    path_f = path if is_quick else scanner_refl_fix(path)

    image_f_pp = cctiff(input_profile, path_f, output_folder)

    # filename = os.path.basename(image_f_pp)
    # full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    image = cv.imread(image_f_pp, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image[0, 0, 0], np.uint16)
    scalar = 65535 if is_16bit else 255
    image_float = image / scalar
    image_float **= PROPHOTO_GAMMA

    os.remove(image_f_pp)

    # convert BGR->RGB
    image_float = image_float[..., ::-1]

    image_float = white_balance(image_float, config["white_point_xyz"])

    # convert RGB->BGR
    image_float = image_float[..., ::-1]

    image_test = image_float ** (1 / PROPHOTO_GAMMA)
    image_test *= scalar
    dst = np.uint16(image_test) if is_16bit else np.uint8(image_test)
    full_output_path = full_save_path_no_extension + "-mh_wb.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, 1600, cv.IMWRITE_TIFF_YDPI, 1600, cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])

    image_float = center_combined(image_float, offsets_file, calibration_input_file, config["lower_hsv"],
                                  config["upper_hsv"], config["xy_offset"])

    image_test = image_float ** (1 / PROPHOTO_GAMMA)
    image_test *= scalar
    dst = np.uint16(image_test) if is_16bit else np.uint8(image_test)
    full_output_path = full_save_path_no_extension + "-mh_centered.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, 1600, cv.IMWRITE_TIFF_YDPI, 1600, cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])

    image_float = shrink_and_clip(image_float,
                                  config.get("first_sharpen", True),
                                  config["black_point_percentage"],
                                  config["gamma"])

    image_float **= 1 / PROPHOTO_GAMMA
    image_float *= scalar
    dst = np.uint16(image_float) if is_16bit else np.uint8(image_float)
    full_output_path = full_save_path_no_extension + "-mh.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, 295, cv.IMWRITE_TIFF_YDPI, 295, cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])

    srgb_and_corners_pipeline(full_output_path, output_folder)

    # os.remove(full_output_path)


def parse_and_validate(args):
    ret = {}
    if args.expansion and args.holo_type:
        ret |= get_config(args.expansion, args.holo_type)
    elif not args.expansion and not args.holo_type:
        # TODO: allow for defining all config through command-line. maybe.
        pass
    else:
        raise ValueError("if either expansion or holo_type is defined, both must be present")

    if args.src_wp:
        source_wp_xyz = np.array(args.src_wp.split(","), dtype=np.float_)
        if len(source_wp_xyz) != 3:
            raise ValueError("source white point must be given as X,Y,Z")
        ret["white_point_xyz"] = source_wp_xyz

    print("config", ret)

    config_keys = ("xy_offset", "white_point_xyz", "black_point_percentage", "gamma", "lower_hsv", "upper_hsv")
    for key in config_keys:
        if key not in ret:
            raise ValueError(key + " must be present")

    return ret


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
        "-e",
        dest="expansion",
        type=str,
        help="expansion abbreviation")
    parser.add_argument(
        "-t",
        dest="holo_type",
        type=str,
        help="holo type (one of nonholo, holo, ex, shattered)")
    parser.add_argument(
        "-i", dest="calibration_input_file", type=str,
        help="input file that describes a calibration created with camera_calibrate -c",
        required=True)
    parser.add_argument(
        "-o", dest="output_folder", type=str, default=".",
        help="output folder for color-corrected images")
    parser.add_argument(
        "-f", dest="offsets_file", type=str,
        help="file that describes offsets of the distortion target and image ROIs")
    parser.add_argument(
        "-p",
        dest="input_profile",
        type=str,
        help="input icc profile",
        required=True)
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="enable quick mode. this skips the call to srf.")
    parser.add_argument(
        "-s",
        dest="src_wp",
        type=str,
        help="source white point as X,Y,Z (scaled to 1). overwrites any config found by specifying -e & -t")
    args = parser.parse_args()
    config = parse_and_validate(args)
    paths = []
    for i in args.path:
        image_path = glob.glob(i)
        if isinstance(image_path, list) and image_path:
            paths.extend(image_path)
        else:
            print(f"{i} is not a valid path, skipping")
    for image_path in paths:
        run_pipeline(image_path, args.input_profile, config, args.calibration_input_file, args.offsets_file,
                     args.output_folder, args.quick)
