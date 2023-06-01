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


HL_WP_NONHOLO = np.array([0.767805, 0.798657, 0.640379])
HL_WP_HOLO = np.array([0.689367, 0.713257, 0.587346])

# all HSV values are from border on ProPhoto g1.8 "as-encoded" RGB values, after white-balancing
# To get these, can open in GIMP, keeping the profile as sRGB, and use eyedropper to get HSV tuples.

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
