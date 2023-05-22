#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""
It is not possible to set exact pixel values for the selection rectangle (aka "rectangle of interest" or roi) directly
in Epson Scan UI. We can edit the plist file (on macOS), but this contains several values, all of which are calculated
from the pixel values of the roi. This utility does the needed math and generates the appropriate plist snippet.
"""

import argparse

import numpy as np


CONFIG_DPI = 1600
# Probably related to DPI, but idk how.
CONFIG_X_PREVIEW_SCALE = 21.350078492935637
# Probably related to DPI, but idk how.
CONFIG_Y_PREVIEW_SCALE = 21.345495661145062
CONFIG_ZOOM_SCALE = 1.1


def print_epson_scan_values(rect_pixel):
    [x, y, w, h] = rect_pixel
    # original_x = 7395.1455479452043
    # original_y = 13865.29094330252
    # original_w = 5692.854452054793
    # original_h = 4134.7087632592284
    print("rect_pixel", rect_pixel)
    rect_inch = rect_pixel / CONFIG_DPI
    print("rect_inch", rect_inch)
    # rect_inch*25.4 may be closer to what Epson Scan does.
    rect_mm = rect_pixel * (25.4/CONFIG_DPI)
    print("rect_mm", rect_mm)
    print("rect_mm_from_in", rect_inch*25.4)
    print("rect_preview_pixel ratios", [
        7395.1455479452043/346.37556720890404,
        13865.29094330252/649.5651899309762,
        5692.854452054793/266.64325632050759,
        4134.7087632592284/193.70404083825454
    ])
    # Best guess is that this is for the roi in the gui itself.
    # Not sure about these calcs; could be a rounded value is scaled instead.
    rect_preview_pixel = [
        x/CONFIG_X_PREVIEW_SCALE,
        y/CONFIG_Y_PREVIEW_SCALE,
        w/CONFIG_X_PREVIEW_SCALE,
        h/CONFIG_Y_PREVIEW_SCALE
    ]
    print("rect_preview_pixel", rect_preview_pixel)

    # rect_zoom is a larger roi around the original roi that is relevant only for the Zoom feature.
    # Nevertheless, we probably want to set these values to reasonable ones so the app doesn't get into a bad state.
    # w and/or h of the zoom roi are scaled from the originals, but the scaling seems to vary.
    # Sometimes they are the same scale. Sometimes they aren't.
    # Let's just use 1.1.
    w_zoom = w*CONFIG_ZOOM_SCALE
    h_zoom = h*CONFIG_ZOOM_SCALE
    rect_zoom_pixel = np.array([
        x - (w_zoom-w)/2,
        y - (h_zoom-h)/2,
        w_zoom,
        h_zoom
    ], dtype=np.float64)
    print("rect_zoom_pixel", rect_zoom_pixel)
    rect_zoom_inch = rect_zoom_pixel / CONFIG_DPI
    print("rect_zoom_inch", rect_zoom_inch)
    rect_zoom_mm = rect_zoom_pixel * (25.4/CONFIG_DPI)
    print("rect_zoom_mm", rect_zoom_mm)
    print("----------------------------------------------------------")

    output = f"""
							<key>marqueeRectInfo</key>
							<dict>
								<key>rect_inch</key>
								<string>{{{{{rect_inch[0]}, {rect_inch[1]}}}, {{{rect_inch[2]}, {rect_inch[3]}}}}}</string>
								<key>rect_mm</key>
								<string>{{{{{rect_mm[0]}, {rect_mm[1]}}}, {{{rect_mm[2]}, {rect_mm[3]}}}}}</string>
								<key>rect_pixel</key>
								<string>{{{{{rect_pixel[0]}, {rect_pixel[1]}}}, {{{rect_pixel[2]}, {rect_pixel[3]}}}}}</string>
								<key>rect_preview_pixel</key>
								<string>{{{{{rect_preview_pixel[0]}, {rect_preview_pixel[1]}}}, {{{rect_preview_pixel[2]}, {rect_preview_pixel[3]}}}}}</string>
								<key>rect_zoom_inch</key>
								<string>{{{{{rect_zoom_inch[0]}, {rect_zoom_inch[1]}}}, {{{rect_zoom_inch[2]}, {rect_zoom_inch[3]}}}}}</string>
								<key>rect_zoom_mm</key>
								<string>{{{{{rect_zoom_mm[0]}, {rect_zoom_mm[1]}}}, {{{rect_zoom_mm[2]}, {rect_zoom_mm[3]}}}}}</string>
								<key>rect_zoom_pixel</key>
								<string>{{{{{rect_zoom_pixel[0]}, {rect_zoom_pixel[1]}}}, {{{rect_zoom_pixel[2]}, {rect_zoom_pixel[3]}}}}}</string>
							</dict>
							<key>outputSizeInfo</key>
							<dict>
								<key>ESPreviewFixOutputSize</key>
								<false/>
								<key>ESPreviewOutputFlipHorizontal</key>
								<false/>
								<key>ESPreviewOutputHeight</key>
								<real>{h}</real>
								<key>ESPreviewOutputWidth</key>
								<real>{w}</real>
								<key>heightInch</key>
								<real>{rect_inch[3]}</real>
								<key>heightMM</key>
								<real>{rect_mm[3]}</real>
								<key>widthInch</key>
								<real>{rect_inch[2]}</real>
								<key>widthMM</key>
								<real>{rect_mm[2]}</real>
							</dict>

--------------------------------------------------------------
				<key>ESPreviewOutputHeight</key>
				<real>{h}</real>
				<key>ESPreviewOutputWidth</key>
				<real>{w}</real>

    """
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Epson settings from desired pixel offset and area')
    parser.add_argument(
        "roi",
        metavar="input_image",
        type=str,
        nargs="+",
        help="x offset, y offset, width, height. All values are floats and relative to the top-left corner of a single roi (rectangle of interest). Assumes 1600 dpi.")
    args = parser.parse_args()
    np.set_printoptions(precision=80)
    if (len(args.roi) != 4):
        raise ValueError("Exactly 4 params are needed to define the roi.")
    [x, y, w, h] = args.roi
    rect_pixel = np.array(args.roi, dtype=np.float64)
    print_epson_scan_values(rect_pixel)
