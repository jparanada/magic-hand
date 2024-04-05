#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import glob
import os

import numpy as np


UNCROPPED_OUTPUT_WIDTH = 754
UNCROPPED_OUTPUT_HEIGHT = 1044
OUTPUT_DIMS = (UNCROPPED_OUTPUT_WIDTH, UNCROPPED_OUTPUT_HEIGHT)


def convert_hsv_to_opencv_hsv(hsv):
    return np.array([hsv[0] / 2, hsv[1] / 100 * 255, hsv[2] / 100 * 255])


HL_WP_NONHOLO = np.array([0.767805, 0.798657, 0.640379])
HL_WP_HOLO = np.array([0.689367, 0.713257, 0.587346])

TMB_WP_NONHOLO = np.array([0.780589, 0.806007, 0.674934])
# TMBM2_WP_NONHOLO = np.array([0.764853, 0.795969, 0.617851])
# 0.756714 0.788797 0.611248

# wp for falkner's fearow
TMBM2_WP_NONHOLO = np.array([0.756714, 0.788797, 0.611248])

TMB_WP_NONHOLO_MANTINE = np.array([0.836413, 0.844644, 0.797932])

BW_WP_NONHOLO = np.array([0.749031, 0.773730, 0.668747])
BW_WP_HOLO = 0.84 * BW_WP_NONHOLO

# RGB 57819 57887 58937
SV_WP_NONHOLO = np.array([0.770666, 0.799261, 0.681583])
SV_WP_HOLO = 0.825 * SV_WP_NONHOLO

# all HSV values are from border on ProPhoto g1.8 "as-encoded" RGB values, after white-balancing
# To get these, can open in GIMP, keeping the profile as sRGB, and use eyedropper to get HSV tuples.

# TMB HSVs
# left 57.4 69.6 89.1
# top right 52.5 74.0 79.2
# right 51.4 74.3 80.1
# bottom left 51.1 75.0 83.4
# bottom 57.8 67.3 91.4
LOWER_TMB_HSV_NONHOLO = np.array([48 / 2, 64 / 100 * 255, 78 / 100 * 255])
UPPER_TMB_HSV_NONHOLO = np.array([61 / 2, 78 / 100 * 255, 94 / 100 * 255])

# (231, 224, 105) -> np.array([56.67/2, 54.55/100*255, 90.59/100*255]) dark yellow, bottom
# HSV 57.3 61,2 87.2 EX border dark yellow again
# (220, 226, 160) -> np.array([57.39/2, 30.13/100*255, 89.8/100*255]) pale yellow
LOWER_HSV_EX_NONHOLO = np.array([51 / 2, 25 / 100 * 255, 80 / 100 * 255])
UPPER_HSV_EX_NONHOLO = np.array([61 / 2, 64 / 100 * 255, 100 / 100 * 255])

# for Kyogre ex HL pp wb
# bottom left 210 10 72
# bottom 212 11.5 68
# lower right 209.5 13.4 52.5
# upper left 212.6 13 68.7
# top 212 14 58
# 210 5-17 50-74

# let's go with HSV 200 8.5 55, 193-207 5-12.5 40-64
LOWER_HSV_EX_EX = np.array([203 / 2, 7 / 100 * 255, 40 / 100 * 255])
UPPER_HSV_EX_EX = np.array([217 / 2, 20 / 100 * 255, 70 / 100 * 255])

# 204 10 67
# 71 4 100
# 0 0 100
# 224 15 80
# 214 15 78
LOWER_HSV_EX_REGI = np.array([190 / 2, 6 / 100 * 255, 65 / 100 * 255])
UPPER_HSV_EX_REGI = np.array([230 / 2, 20 / 100 * 255, 85 / 100 * 255])
LOWER_HSV_EX_REGI_2 = np.array([0 / 2, 0 / 100 * 255, 98 / 100 * 255])
UPPER_HSV_EX_REGI_2 = np.array([360 / 2, 12 / 100 * 255, 100 / 100 * 255])

# Wizards yellow: 176 163 39 -> [54.31 / 2, 77.84 / 100 * 255, 69.02 / 100 * 255]
LOWER_HSV_WIZARDS = np.array([49.31 / 2, 72.84 / 100 * 255, 64.02 / 100 * 255])
UPPER_HSV_WIZARDS = np.array([59.31 / 2, 82.84 / 100 * 255, 74.02 / 100 * 255])

# CL is TCG Classic
LOWER_HSV_CL = convert_hsv_to_opencv_hsv([50.8, 43.4, 81.8])
UPPER_HSV_CL = convert_hsv_to_opencv_hsv([60.8, 58.4, 91.8])

# 55.7 47.8 89.0
LOWER_HSV_CL_SWSH = convert_hsv_to_opencv_hsv([50.7, 43.8, 84.0])
UPPER_HSV_CL_SWSH = convert_hsv_to_opencv_hsv([60.7, 53.8, 94.0])

LOWER_HSV_CL_SV = convert_hsv_to_opencv_hsv([190.0, 0.0, 49.0])
UPPER_HSV_CL_SV = convert_hsv_to_opencv_hsv([254.9, 19.0, 76.0])

# a wide range here because the HS border has varying colors. but empirically this works
LOWER_HSV_CL_HS = convert_hsv_to_opencv_hsv([48.0, 23.0, 80.0])
UPPER_HSV_CL_HS = convert_hsv_to_opencv_hsv([68.0, 44.0, 95.0])

# 58.5 48.5 89.6
LOWER_HSV_BW = convert_hsv_to_opencv_hsv([54, 44.0, 85.0])
UPPER_HSV_BW = convert_hsv_to_opencv_hsv([64, 52.0, 93.0])

# 58 49 90 - most of the border
# 61 19 90 - light part
LOWER_HSV_SF = convert_hsv_to_opencv_hsv([54, 35.0, 86.0])
UPPER_HSV_SF = convert_hsv_to_opencv_hsv([64, 53.0, 95.0])


config = {
    "hl": {
        "xy_offset": np.array([0, 0]),
        "nonholo": {
            "white_point_xyz": HL_WP_NONHOLO,
            "black_point_percentage": 17,
            "gamma": 0.78,
            "lower_hsv": LOWER_HSV_EX_NONHOLO,
            "upper_hsv": UPPER_HSV_EX_NONHOLO
        },
        "holo": {
            "white_point_xyz": HL_WP_HOLO,
            "black_point_percentage": 7,
            "gamma": 0.87,
            "lower_hsv": LOWER_HSV_EX_NONHOLO,
            "upper_hsv": UPPER_HSV_EX_NONHOLO
        },
        "ex": {
            "white_point_xyz": HL_WP_HOLO,
            "black_point_percentage": 7,
            "gamma": 0.87,
            "lower_hsv": LOWER_HSV_EX_EX,
            "upper_hsv": UPPER_HSV_EX_EX
        },
        "shattered": {
            # not a mistake; the nonholo white point works best here
            "white_point_xyz": HL_WP_NONHOLO,
            "black_point_percentage": 15.5,
            "gamma": 0.78,
            "lower_hsv": np.array([LOWER_HSV_EX_REGI, LOWER_HSV_EX_REGI_2]),
            "upper_hsv": np.array([UPPER_HSV_EX_REGI, UPPER_HSV_EX_REGI_2])
        },
    },
    "tmb": {
        # "xy_offset": np.array([59, 64]),  # for tropical breeze
        # "xy_offset": np.array([14, -1084]),  # for switch bc my edge detection uses mode :(
        "xy_offset": np.array([125, -361]),
        "nonholo": {
            "white_point_xyz": TMB_WP_NONHOLO,
            "black_point_percentage": 7,
            "gamma": 0.87,
            "lower_hsv": LOWER_TMB_HSV_NONHOLO,
            "upper_hsv": UPPER_TMB_HSV_NONHOLO
        }
    },
    "tmbm2": {
        "xy_offset": np.array([125, -361]),
        "nonholo": {
            "white_point_xyz": TMBM2_WP_NONHOLO,
            "black_point_percentage": 7,
            "gamma": 0.87,
            "lower_hsv": LOWER_TMB_HSV_NONHOLO,
            "upper_hsv": UPPER_TMB_HSV_NONHOLO
        },
        "scale": 5709/5692
    },
    "cl-normal": {
        "xy_offset": np.array([0, 0]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": SV_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_CL,
            "upper_hsv": UPPER_HSV_CL
        }
    },
    "cl-sm-trainer": {
        "xy_offset": np.array([0, -19]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": SV_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_CL,
            "upper_hsv": UPPER_HSV_CL
        }
    },
    "cl-swsh-trainer": {
        "xy_offset": np.array([0, -18]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": SV_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_CL_SWSH,
            "upper_hsv": UPPER_HSV_CL_SWSH
        }
    },
    "cl-sv-trainer": {
        "xy_offset": np.array([0, -17]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": SV_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_CL_SV,
            "upper_hsv": UPPER_HSV_CL_SV
        }
    },
    "cl-hgss": {
        "xy_offset": np.array([0, 0]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": SV_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_CL_HS,
            "upper_hsv": UPPER_HSV_CL_HS
        }
    },
    "bw": {
        "xy_offset": np.array([0, 0]),
        "holo": {
            "first_sharpen": False,
            "white_point_xyz": BW_WP_HOLO,
            "black_point_percentage": 20,
            "gamma": 0.705,
            "lower_hsv": LOWER_HSV_BW,
            "upper_hsv": UPPER_HSV_BW
        }
    },
    # unsure if this extends back to DP, so just calling it Stormfront
    "sf": {
        "xy_offset": np.array([0, 0]),
        "nonholo": {
            "first_sharpen": True,
            "white_point_xyz": BW_WP_HOLO,
            "black_point_percentage": 17,
            "gamma": 0.78,
            "lower_hsv": LOWER_HSV_SF,
            "upper_hsv": UPPER_HSV_SF
        }
    }
}


def get_config(set, holo_type):
    ret = {}
    set_config = config[set]
    ret["xy_offset"] = set_config["xy_offset"]
    return ret | set_config[holo_type]


def config_pipeline(image_path, set, holo_type):
    # TODO: add mode that gets set/holo_type from file name
    print(get_config(set, holo_type))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get config for the given card(s)")
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
        help="expansion abbreviation",
        required=True)
    parser.add_argument(
        "-t",
        dest="holo_type",
        type=str,
        help="holo type (one of nonholo, holo, ex, shattered)",
        required=True)
    args = parser.parse_args()
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
            config_pipeline(i, args.expansion, args.holo_type)
        else:
            raise ValueError("{} does not exist".format(i))
