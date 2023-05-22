#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import glob
import os
from pathlib import Path
import subprocess

import argparse
import cv2 as cv
import numpy as np

CONFIG_SCANNER_REFL_FIX_CALIBRATION_PATH = "/Users/paranada/icc_profiles/scanner/scanner_cal.txt"

PROPHOTO_GAMMA = 1.80078125
SCALAR_16_BIT = 65535
SCALAR_8_BIT = 255


# TODO: this is by far the slowest part of the pipeline. Could implement a "draft" mode that skips it...
# IT IS VERY IMPORTANT THAT THE TIFF BE TAGGED WITH ITS CORRECT DPI, since scanner_refl_fix needs to know the physical
# size of the image to work!
def scanner_refl_fix(path: str) -> str:
    result = subprocess.run([
        "scanner_refl_fix",
        "-C", CONFIG_SCANNER_REFL_FIX_CALIBRATION_PATH,
        "-B", path
    ])
    return "_f".join([os.path.splitext(path)[0], ".tif"]) if result.returncode == 0 else None


"""
cctiff -N -ir ~/profiles/pokemon-colors-202-patch/2022-01-24_v850_202patch_qm_ax_ua.icc -ir ~/common_profiles/LargeRGB-elle-V2-g18.icc in out
"""
def cctiff(input_profile: str, input_image: str, output_path) -> str:
    input_path = Path(input_image)
    output_image = str(Path(output_path).joinpath(input_path.stem + "-ppnp" + input_path.suffix))
    print("output_image", output_image)
    result = subprocess.run([
        "cctiff",
        "-N",
        "-ir", os.path.expanduser(input_profile),
        "-ir", os.path.expanduser("~/Pictures/pokemon-tcg/goods/LargeRGB-elle-V2-g18.icc"),
        input_image,
        output_image
    ])
    return output_image if result.returncode == 0 else None


# only supports pure power functions at the moment (i.e. not piecewise transfer functions)
def read_image_linear_bgr(path: str, power=PROPHOTO_GAMMA) -> np.ndarray:
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image[0, 0, 0], np.uint16)
    scalar = SCALAR_16_BIT if is_16bit else SCALAR_8_BIT
    return cv.pow(image/scalar, power)


def run_pipeline(path: str, input_profile, output_folder) -> None:
    path_f = scanner_refl_fix(path)
    print(path_f)
    image_f_pp = cctiff(input_profile, path_f, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix color for a raw scan with scanner_refl_fix. Outputs a "
                                                 "ProPhoto image.")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    parser.add_argument(
        "-o", dest="output_folder", type=str, default=".",
        help="output folder for color-corrected images")
    parser.add_argument(
        "-p",
        dest="input_profile",
        type=str,
        help="input icc profile",
        required=True)

    args = parser.parse_args()
    paths = []
    for i in args.path:
        image_path = glob.glob(i)
        if isinstance(image_path, list) and image_path:
            paths.extend(image_path)
        else:
            print(f"{i} is not a valid path, skipping")
    for image_path in paths:
        run_pipeline(image_path, args.input_profile, args.output_folder)
