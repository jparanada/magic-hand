#!/usr/bin/env python3
# Copyright 2022 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import glob
import os

import argparse
import colour
import cv2 as cv
from PIL import Image
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure

CONFIG_IMAGE_DPI = 1600
CONFIG_OUTPUT_DIMS = (754, 1044)
CONFIG_CROP_FROM_EDGE = 10

# defined by libtiff
COMPRESSION_NONE = 1
PROPHOTO_GAMMA = 1.80078125
PROPHOTO_XW = 0.964295676
PROPHOTO_YW = 1.0
PROPHOTO_ZW = 0.825104603
PROPHOTO_WP = np.array([PROPHOTO_XW, PROPHOTO_YW, PROPHOTO_ZW])
PROPHOTO_ILLUMINANT_XY = np.array([PROPHOTO_XW / (PROPHOTO_XW + PROPHOTO_YW + PROPHOTO_ZW),
                                   PROPHOTO_YW / (PROPHOTO_XW + PROPHOTO_YW + PROPHOTO_ZW)])


# https://stackoverflow.com/questions/39792163/vectorizing-srgb-to-linear-conversion-properly-in-numpy
def linear_rgb_to_lstar_rgb(image_32: np.ndarray) -> np.ndarray:
    mask = image_32 >= (2700 / 24389 * 0.08)
    image_32[mask] = 1.16 / 1.0 * (image_32[mask] ** (1 / 3) - 0.16 / 1.16)
    image_32[~mask] = image_32[~mask] * 24389 / 2700


def lstar_rgb_to_linear_rgb(image_32: np.ndarray) -> np.ndarray:
    mask = image_32 >= 0.08
    image_32[mask] = (1.0 / 1.16 * image_32[mask] + 0.16 / 1.16) ** 3
    image_32[~mask] = image_32[~mask] * 2700 / 24389


def unsharpen_channel(image, radius, amount, threshold) -> np.ndarray:
    # https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/filters/_unsharp_mask.py#L19-L141
    blurred = ndi.gaussian_filter(image, sigma=radius, mode="reflect", cval=0, truncate=4.0)

    blur_difference = image - blurred
    image_sharpened = image + blur_difference * amount
    if threshold <= 0:
        result = image_sharpened
    else:
        # https://gist.github.com/mlashcorp/1641134
        mask = np.absolute(blur_difference) >= threshold
        result = np.where(mask, image_sharpened, image)
    return np.clip(result, 0., 1., out=result)


def get_matrix_rgb_to_xyz():
    [XW, YW, ZW] = PROPHOTO_WP
    # these are the x, y (lowercase!) of r/g/b primaries for the WS profile from elle's source code.
    # ProPhoto
    (xr, yr) = (0.7347, 0.2653)
    (xg, yg) = (0.1596, 0.8404)
    (xb, yb) = (0.0366, 0.0001)

    Xr = xr / yr
    Yr = 1
    Zr = (1 - xr - yr) / yr
    Xg = xg / yg
    Yg = 1
    Zg = (1 - xg - yg) / yg
    Xb = xb / yb
    Yb = 1
    Zb = (1 - xb - yb) / yb
    [Sr, Sg, Sb] = np.matmul(np.linalg.inv([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ]), [XW, YW, ZW])

    matrix_rgb_to_xyz = np.array([
        [Sr * Xr, Sg * Xg, Sb * Xb],
        [Sr * Yr, Sg * Yg, Sb * Yb],
        [Sr * Zr, Sg * Zg, Sb * Zb]
    ])
    return matrix_rgb_to_xyz


def convert_linearized_prophoto_to_cielab(linearized_rgb: np.ndarray) -> np.ndarray:
    # https://colour.readthedocs.io/en/develop/generated/colour.RGB_to_XYZ.html
    matrix_rgb_to_xyz = get_matrix_rgb_to_xyz()
    XYZ = colour.algebra.vector_dot(matrix_rgb_to_xyz, linearized_rgb)
    return colour.XYZ_to_Lab(XYZ, PROPHOTO_ILLUMINANT_XY)


def convert_cielab_to_linearized_prophoto(cielab: np.ndarray) -> np.ndarray:
    XYZ = colour.Lab_to_XYZ(cielab, PROPHOTO_ILLUMINANT_XY)
    matrix_xyz_to_rgb = np.linalg.inv(get_matrix_rgb_to_xyz())
    rgb = colour.algebra.vector_dot(matrix_xyz_to_rgb, XYZ)

    return np.clip(rgb, 0, 1, out=rgb)


# assume no negative values in the image
def sharpen_lstar_of_cielab(lstar: np.ndarray, radius=1.272, amount=1.0, threshold=1.0) -> np.ndarray:
    scale_factor = 100  # should be the max possible value in the 2d array. for L* this is 100
    return unsharpen_channel(
        lstar / scale_factor, radius=radius, amount=amount, threshold=threshold / scale_factor) * scale_factor


def sharpen_luminosity(image: np.ndarray, radius=1.272, amount=1.0, threshold=1.0) -> np.ndarray:
    image_cielab = convert_linearized_prophoto_to_cielab(image)
    lstar = image_cielab[..., 0]
    lstar = sharpen_lstar_of_cielab(lstar, radius=radius, amount=amount, threshold=threshold)
    image_cielab[..., 0] = lstar
    return convert_cielab_to_linearized_prophoto(image_cielab)


def resize_opencv_bicubic(image: np.ndarray):
    arr_out = cv.resize(image, (CONFIG_OUTPUT_DIMS[0], CONFIG_OUTPUT_DIMS[1]), interpolation=cv.INTER_CUBIC)
    arr_out = np.clip(arr_out, 0., 1., out=arr_out)
    # luminosity sharpening second pass. may have to re-tune params if you change the interpolation method
    arr_out = sharpen_luminosity(arr_out, radius=0.55, amount=0.4, threshold=1.0)
    return arr_out


def resize_opencv_area(image: np.ndarray):
    # opencv doc for resize recommends INTER_AREA for shrinking
    # looks alright, but text is a bit soft
    arr_out = cv.resize(image, (CONFIG_OUTPUT_DIMS[0], CONFIG_OUTPUT_DIMS[1]), interpolation=cv.INTER_AREA)
    arr_out = np.clip(arr_out, 0., 1., out=arr_out)
    # luminosity sharpening second pass. may have to re-tune params if you change the interpolation method
    arr_out = sharpen_luminosity(arr_out, radius=0.55, amount=0.75, threshold=1.0)
    return arr_out


def resize_pillow(image: np.ndarray):
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters says BICUBIC, LANCZOS have best
    # downscaling quality. BICUBIC looks good based on my own experiments, although there is some ringing around text
    resample_filter = Image.Resampling.BICUBIC
    resized_channels = []
    for i in range(image.shape[2]):
        channel = Image.fromarray(image[:, :, i])
        channel = channel.resize(CONFIG_OUTPUT_DIMS, resample=resample_filter)
        resized_channels.append(channel)
    arr_out = np.stack(resized_channels, axis=2)
    arr_out = np.clip(arr_out, 0, 1, out=arr_out)
    # luminosity sharpening second pass. may have to re-tune params if you change the resample filter
    return sharpen_luminosity(arr_out, radius=0.55, amount=0.5, threshold=1.0)


def shrink_and_clip(image_float, black_point_percentage=7, gamma=0.87):
    # convert BGR->RGB
    image_float = image_float[..., ::-1]

    image_cielab = convert_linearized_prophoto_to_cielab(image_float)
    lstar = image_cielab[..., 0]

    # luminosity contrast
    lstar = exposure.rescale_intensity(lstar, in_range=(black_point_percentage, 100), out_range=(0, 100))
    lstar /= 100
    lstar = np.power(lstar, gamma, out=lstar)
    lstar *= 100

    # luminosity sharpening first pass
    lstar = sharpen_lstar_of_cielab(lstar, radius=1.4, amount=1.2, threshold=1.0)

    image_cielab[..., 0] = lstar
    image_float = convert_cielab_to_linearized_prophoto(image_cielab)

    image_float = resize_opencv_bicubic(image_float)

    # crop
    image_float = image_float[
               CONFIG_CROP_FROM_EDGE:CONFIG_OUTPUT_DIMS[1]-CONFIG_CROP_FROM_EDGE,
               CONFIG_CROP_FROM_EDGE:CONFIG_OUTPUT_DIMS[0]-CONFIG_CROP_FROM_EDGE,
               :]

    # convert RGB->BGR
    return image_float[..., ::-1]


def shrink_and_clip_pipeline(img_file: str, save_path: str):
    filename = os.path.basename(img_file)
    # full_save_path = os.path.join(save_path, filename)
    full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    image = cv.imread(img_file, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image[0][0][0], np.uint16)
    scalar = 65535 if is_16bit else 255

    image_float = image / scalar
    image_float **= PROPHOTO_GAMMA

    shrink_and_clip(image_float)

    # we're writing the file out just to do color conversion on it, feelsbadman
    # since pillow (and therefore ImageCms) doesn't support 3-channel 16-bit images
    image_float **= 1 / PROPHOTO_GAMMA
    image_float *= scalar
    dst = np.uint16(image_float) if is_16bit else np.uint8(image_float)
    full_output_path = full_save_path_no_extension + "-cropped.tif"
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, CONFIG_IMAGE_DPI, cv.IMWRITE_TIFF_YDPI, CONFIG_IMAGE_DPI,
                cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sharpen, downscale, and crop an image (not including corner-rounding)")
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
        help="output folder for shrunk and cropped images",
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
            shrink_and_clip_pipeline(i, args.output_folder)
        else:
            raise ValueError("{} does not exist".format(i))
