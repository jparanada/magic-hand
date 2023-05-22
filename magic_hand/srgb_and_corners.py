#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import glob
import os
import subprocess

from PIL import Image


MASK_PATH = os.path.expanduser("~/Pictures/pokemon-tcg/card_mask_rounded_4.761905_radius.png")
RGB_WHITE = (255, 255, 255)


# replaces the input file with a losslessly-optimized version
def oxipng(path: str) -> str:
    result = subprocess.run([
        "oxipng",
        "-o", "4",
        "--strip", "all",
        path
    ])
    return path if result.returncode == 0 else None


def cctiff_srgb(path: str) -> str:
    output_path = "_final".join(os.path.splitext(path))
    result = subprocess.run([
        "cctiff",
        "-N",
        "-ir", os.path.expanduser("~/common_profiles/LargeRGB-elle-V2-g18.icc"),
        "-ir", os.path.expanduser("~/common_profiles/windows/sRGB.icc"),
        path,
        output_path
    ])
    return output_path if result.returncode == 0 else None


def srgb_and_corners_pipeline(image_path, output_folder):
    image_path_srgb = cctiff_srgb(image_path)

    # We can open this 3-channel 16 bpc tiff okay in pillow, it just auto-converts to 8 bpc.
    # This is fine because we are going to export as 8 bpc anyway.
    card = Image.open(image_path_srgb)
    os.remove(image_path_srgb)
    alpha = Image.open(MASK_PATH).convert("L")
    card.putalpha(alpha)

    filename = os.path.basename(image_path_srgb)
    full_save_path_no_extension = os.path.join(output_folder, os.path.splitext(filename)[0])
    png_path = full_save_path_no_extension + ".png"
    card.save(png_path)
    oxipng(png_path)

    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    card_for_jpg = Image.new("RGB", card.size, RGB_WHITE)
    # Could use card.getchannel("A") for the mask if we didn't already have the alpha channel
    card_for_jpg.paste(card, mask=alpha)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
    card_for_jpg.save(full_save_path_no_extension + ".jpg", quality=75, subsampling="4:4:4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sharpen, downscale, and crop an image")
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
            srgb_and_corners_pipeline(i, args.output_folder)
        else:
            raise ValueError("{} does not exist".format(i))
