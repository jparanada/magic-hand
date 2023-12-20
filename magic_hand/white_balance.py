#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import glob
import os

import colour
import cv2 as cv
import numpy as np


CONFIG_IMAGE_DPI = 1600

PROPHOTO_GAMMA = 1.80078125
PROPHOTO_WP = np.array([0.964295676, 1.0, 0.825104603])

REC2020_WP = np.array([0.95045471, 1.0, 1.08905029])

# defined by libtiff
COMPRESSION_NONE = 1

"""
% xicclu -s65535 -ia -py /Users/paranada/profiles/pokemon-colors-202-patch/2022-01-24_v850_202patch_qm_al_ua.icc
55920 56366 54847
55920.000000 56366.000000 54847.000000 [RGB] -> Lut -> 0.825164 0.345817 0.358748 [Yxy]
"""
def white_balance(linearized_rgb, illuminant_src):
    # my own measured marill wp in scanner profile RGB: 55994 56437 54901
    # xicclu -s65535 -ia -px /Users/paranada/profiles/pokemon-colors-202-patch/2022-01-24_v850_202patch_qm_al_ua.icc
    # 55994.000000 56437.000000 54901.000000 [RGB] -> Lut -> 0.796966 0.825067 0.681396 [XYZ]

    # 1. use colprof -ua, attach that profile to your image, relcol or abscol convert to desired working space
    # (call this the WS profile)
    # relcol vs abscol won't matter there (it'd be identical--that's what `-ua` does). export the image w/NO profile
    # attached--this is important since GIMP only reads Pixel values with eyedropper correctly if it thinks the image is
    # sRGB.
    # 2. in GIMP load the image, eyedropper the desired white, note the Pixel tuple.
    # 3. xicclu -s65535 -ia -px /path/to/WS-profile.icc
    #  and then look up the Pixel tuple from #2
    # this tuple is illuminant_src.
    illuminant_dst = PROPHOTO_WP  # REC2020_WP

    # clip white less aggressively by scaling the white point back a bit
    illuminant_dst *= 0.995

    [XW, YW, ZW] = PROPHOTO_WP
    # these are the x, y (lowercase!) of r/g/b primaries for the WS profile from elle's source code.
    # ProPhoto
    (xr, yr) = (0.7347, 0.2653)
    (xg, yg) = (0.1596, 0.8404)
    (xb, yb) = (0.0366, 0.0001)

    # Rec 2020
    # (xr, yr) = (0.708012540607, 0.291993664388)
    # (xg, yg) = (0.169991652439, 0.797007778423)
    # (xb, yb) = (0.130997824007, 0.045996550894)

    Xr = xr/yr
    Yr = 1
    Zr = (1-xr-yr)/yr
    Xg = xg/yg
    Yg = 1
    Zg = (1-xg-yg)/yg
    Xb = xb/yb
    Yb = 1
    Zb = (1-xb-yb)/yb
    [Sr, Sg, Sb] = np.matmul(np.linalg.inv([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ]), [XW, YW, ZW])

    matrix_rgb_to_xyz = np.array([
        [Sr*Xr, Sg*Xg, Sb*Xb],
        [Sr*Yr, Sg*Yg, Sb*Yb],
        [Sr*Zr, Sg*Zg, Sb*Zb]
    ])
    # print("my matrix_rgb_to_xyz\n", matrix_rgb_to_xyz)
    # bruce_matrix_rgb_to_xyz = np.array([
    #     [0.7976749, 0.1351917, 0.0313534],
    #     [0.2880402, 0.7118741, 0.0000857],
    #     [0.0000000, 0.0000000, 0.8252100]
    # ])
    # print("bruce_matrix_rgb_to_xyz\n", bruce_matrix_rgb_to_xyz)
    chromatic_adaptation_transform = "Bradford"

    # https://colour.readthedocs.io/en/develop/generated/colour.RGB_to_XYZ.html
    XYZ = colour.algebra.vector_dot(matrix_rgb_to_xyz, linearized_rgb)
    # print("XYZ pre-transform\n", XYZ)
    if chromatic_adaptation_transform is not None:
        # https://colour.readthedocs.io/en/develop/_modules/colour/adaptation/vonkries.html#matrix_chromatic_adaptation_VonKries
        M_CAT = colour.adaptation.matrix_chromatic_adaptation_VonKries(
            illuminant_src,
            illuminant_dst,
            transform=chromatic_adaptation_transform,
        )
        XYZ = colour.algebra.vector_dot(M_CAT, XYZ)
        # print(f"M_CAT for {chromatic_adaptation_transform}\n", M_CAT)
    white_balanced_xyz = XYZ
    # print("XYZ post-transform\n", white_balanced_xyz)

    (h, w, _) = np.shape(white_balanced_xyz)
    matrix_xyz_to_rgb = np.linalg.inv(matrix_rgb_to_xyz)
    # print("my matrix_xyz_to_rgb\n", matrix_xyz_to_rgb)

    white_balanced_rgb = colour.algebra.vector_dot(
        matrix_xyz_to_rgb,
        white_balanced_xyz)

    res = np.where(white_balanced_rgb > 1.0, True, False)
    res = np.any(res, axis=2)
    rgb_overflow_count_fast = np.sum(res)
    print("rgb overflow count", rgb_overflow_count_fast)

    # print("clipped rgb\n", np.clip(white_balanced_rgb, 0, 1))
    # raise Exception("test only")
    return np.clip(white_balanced_rgb, 0, 1, white_balanced_rgb)


def white_balance_pipeline(img_file, source_wp_xyz, save_path):
    np.set_printoptions(precision=80, suppress=True)

    filename = os.path.basename(img_file)
    full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    image = cv.imread(img_file, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image[0, 0, 0], np.uint16)
    scalar = 65535 if is_16bit else 255
    image = cv.pow(image/scalar, PROPHOTO_GAMMA)

    # convert BGR->RGB
    # start at channel 2 to omit alpha (channel 3) if it's there
    image = image[..., 2::-1]

    image = white_balance(image, source_wp_xyz)

    # convert RGB->BGR
    image = image[..., ::-1]

    if is_16bit:
        dst = np.uint16(scalar * cv.pow(image, 1/PROPHOTO_GAMMA))
    else:
        dst = np.uint8(scalar * cv.pow(image, 1/PROPHOTO_GAMMA))
    full_output_path = full_save_path_no_extension + "-whitebal.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, CONFIG_IMAGE_DPI, cv.IMWRITE_TIFF_YDPI, CONFIG_IMAGE_DPI,
                cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="white balance an image")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    parser.add_argument(
        "-s",
        dest="src_wp",
        type=str,
        help="source white point as X,Y,Z (scaled to 1)",
        required=True)
    parser.add_argument(
        "-o",
        dest="output_folder",
        type=str,
        help="output folder for images",
        required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.output_folder):
        raise ValueError("Output path is not a folder")
    source_wp_xyz = np.array(args.src_wp.split(","), dtype=np.float_)
    if len(source_wp_xyz) != 3:
        raise ValueError("source white point must be given as X,Y,Z")
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
            white_balance_pipeline(i, source_wp_xyz, args.output_folder)
        else:
            raise ValueError("{} does not exist".format(i))
