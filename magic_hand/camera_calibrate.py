#!/usr/bin/env python3
# Copyright 2023 Jonathan Paranada
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import glob
import json
import os

import argparse
import cv2 as cv
import numpy as np
import skimage.transform as transform

CONFIG_NUM_COLUMN_CORNERS = 34  #28  # 34
CONFIG_NUM_ROW_CORNERS = 34  #20  # 34
CONFIG_USE_CENTER_DOTS = False
CONFIG_USE_SB = True
CONFIG_SQUARE_LENGTH_INCHES = 3/25.4
CONFIG_TARGET_DPI = 1600
CONFIG_IMAGE_DPI = 1600
CONFIG_SHOULD_ROTATE = False

# defined by libtiff
COMPRESSION_NONE = 1
PROPHOTO_GAMMA = 1.80078125

# TODO: these should be passed in
CONFIG_X_TARGET = 1
CONFIG_Y_TARGET = 1
CONFIG_W_TARGET = 1
CONFIG_H_TARGET = 1
CONFIG_X_IMAGE = 1
CONFIG_Y_IMAGE = 1
CONFIG_W_IMAGE = 1
CONFIG_H_IMAGE = 1


def find_corners(color):
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)

    imgpoints = [] # 2d points in image plane.

    # cv.imshow("gray", gray)
    # cv.waitKey(0)

    print("about to find corners")

    flags = 0
    if CONFIG_USE_CENTER_DOTS:
        flags += cv.CALIB_CB_MARKER

    # Find the chessboard corners
    if CONFIG_USE_SB:
        ret, corners = cv.findChessboardCornersSB(
            gray, (CONFIG_NUM_COLUMN_CORNERS, CONFIG_NUM_ROW_CORNERS), None, flags)
    else:
        ret, corners = cv.findChessboardCorners(
            gray, (CONFIG_NUM_COLUMN_CORNERS, CONFIG_NUM_ROW_CORNERS), None, flags)

    if ret:
        # Workaround for https://github.com/opencv/opencv/issues/22083.
        # Note sometimes the corners are in a different order, and
        # other times they match findChessboardCorners.
        # It varies image-by-image.
        if CONFIG_USE_SB:
            # The "good" ordering is if the first corner is closest to
            # upper-left.  We could make this check more robust by
            # looking at all four corners...
            if np.linalg.norm(corners[0][0] - [0, 0]) > np.linalg.norm(corners[-1][0] - [0, 0]):
                print("rearranging sb corners to match non-sb")
                new_corners = np.flipud(
                    corners[
                        -1
                        :CONFIG_NUM_COLUMN_CORNERS*CONFIG_NUM_ROW_CORNERS
                            - CONFIG_NUM_COLUMN_CORNERS - 1
                        :-1]).tolist()
                # print(new_corners)
                for i in range(1, CONFIG_NUM_ROW_CORNERS+1):
                    new_corners.extend(
                        corners[
                            -i*CONFIG_NUM_COLUMN_CORNERS
                            :-i*CONFIG_NUM_COLUMN_CORNERS + CONFIG_NUM_COLUMN_CORNERS])
                corners = np.array(new_corners, dtype=np.float32)
            else:
                print("no need to rearrange sb corners")
        else:
            print("didn't use sb, so refining corners with cornerSubPix")
            # Are these criteria/numbers/params good for 1600 dpi
            # images? Man, who even knows.
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # for corner in corners:
        #     print(corner)

        imgpoints.append(corners)
        # Draw and display the corners
        color_display = copy.deepcopy(color)
        cv.drawChessboardCorners(color_display, (CONFIG_NUM_COLUMN_CORNERS,CONFIG_NUM_ROW_CORNERS),
            corners, ret)
        # cv.imshow("img", color_display)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    else:
        raise ValueError(
            "No corners found! Check that the configured number of rows and columns is exactly"
            " correct")
    return imgpoints


def calibrate(img):
    # https://docs.opencv.org/4.6.0/dc/dbb/tutorial_py_calibration.html
    imgpoints = find_corners(img)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CONFIG_NUM_ROW_CORNERS*CONFIG_NUM_COLUMN_CORNERS,3), np.float32)
    objp[:,:2] = np.mgrid[0:CONFIG_NUM_COLUMN_CORNERS,0:CONFIG_NUM_ROW_CORNERS].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    objpoints.append(objp)
    h = img.shape[0]
    w = img.shape[1]

    ret, camera_mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None)
    if not ret:
        raise ValueError("calibrateCamera failed!")

    print_error(objpoints, imgpoints, rvecs, tvecs, camera_mtx, dist)

    r_matrix, _ = cv.Rodrigues(rvecs[0])
    # print(f"camera_mtx: {camera_mtx}")
    # print(f"dist: {dist}")
    # print(f"r_matrix: {r_matrix}")
    # print(f"tvecs[0]: {tvecs[0]}")

    return ret, camera_mtx, dist, r_matrix, tvecs[0]


def undistort(img, mtx, distortion_coeffs):
    h = img.shape[0]
    w = img.shape[1]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, distortion_coeffs, (w, h), 1, (w, h))
    # print(f"roi: {roi}")

    mapx, mapy = cv.initUndistortRectifyMap(mtx, distortion_coeffs, None, newcameramtx, (w, h), cv.CV_32FC1)
    """
    opencv remap uses quantized bilinear interpolation.
    See https://stackoverflow.com/questions/42880897/opencv-remap-interpolation-error/42881329#42881329 for details
    and https://www.crisluengo.net/archives/1140/ also.
    remap does not support exact bilinear interpolation, per the doc.
    However, no visible difference vs (unused) undistort_skimage below, and that is much slower.
    """
    # https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function for explanation of how maps work
    dst = cv.remap(img, mapx, mapy, interpolation=cv.INTER_LINEAR)
    print("end remap.")

    # cv.imshow("undistort no perspective fix", dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Can crop the image to roi.  But for simplicity's sake let's not,
    # so that output size = input size.
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst, mapx, mapy


def undistort_skimage(img, mtx, distortion_coeffs):
    # print("undistort")
    h = img.shape[0]
    w = img.shape[1]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, distortion_coeffs, (w, h), 1, (w, h))
    # print(f"roi: {roi}")

    mapx, mapy = cv.initUndistortRectifyMap(mtx, distortion_coeffs, None, newcameramtx, (w, h), cv.CV_32FC1)
    print("mapx shape", mapx.shape)
    print("mapx", mapx)
    print("mapy shape", mapy.shape)
    print("mapy", mapy)

    inverse_map = np.array([mapy, mapx])
    print("inverse_map shape", inverse_map.shape)
    dst_sk_b = transform.warp(img[:, :, 0], inverse_map, order=1, clip=False, preserve_range=True)
    dst_sk_g = transform.warp(img[:, :, 1], inverse_map, order=1, clip=False, preserve_range=True)
    dst_sk_r = transform.warp(img[:, :, 2], inverse_map, order=1, clip=False, preserve_range=True)
    dst_sk = np.stack((dst_sk_b, dst_sk_g, dst_sk_r), axis=2)
    print("dst_sk shape", dst_sk.shape)
    dst_sk = np.clip(dst_sk, 0, 1, dst_sk)
    # cv.imshow("dst_sk", dst_sk)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return dst_sk

# THIS CODE IS WRONG because the maps are for which coordinate in the input maps to an integer index in the map.
# Thus doing matrix multiplication on the maps is not correct.
# def undistort_one_shot(img, mapx, mapy, affine_matrix, perspective_matrix):
#     map_reshaped = np.stack((mapx, mapy, np.ones(mapx.shape, dtype=np.float32)), axis=2)
#     print("mapx type", mapx.dtype)
#     print("map_reshaped shape", map_reshaped.shape)
#     # combined_matrix = np.linalg.inv(affine_matrix) @ np.linalg.inv(perspective_matrix)
#     combined_matrix = np.linalg.inv(affine_matrix @ perspective_matrix)
#     complete_transform = np.float32(np.einsum("ij,...j", combined_matrix, map_reshaped))
#     complete_transform_mapx = complete_transform[..., 0]
#     print("complete_transform_mapx type", complete_transform_mapx.dtype)
#     complete_transform_mapy = complete_transform[..., 1]
#     img_transformed = cv.remap(img, complete_transform_mapx, complete_transform_mapy, interpolation=cv.INTER_LINEAR)
#
#     res = np.where(img_transformed > 1.0, True, False)
#     res = np.any(res, axis=2)
#     rgb_overflow_count_fast = np.sum(res)
#     print("rgb overflow count", rgb_overflow_count_fast)
#
#     return img_transformed


def find_perspective_with_cb_detection(img):
    # We want the four corners to be
    # <number of squares>*CONFIG_SQUARE_LENGTH_INCHES*CONFIG_IMAGE_DPI
    # away from each other.
    # Note we're using the corners inside the board, not the very
    # outermost corners.
    corner_points = find_corners(img)[0]
    tl = corner_points[0][0]
    tr = corner_points[CONFIG_NUM_COLUMN_CORNERS - 1][0]
    br = corner_points[-1][0]
    bl = corner_points[-CONFIG_NUM_COLUMN_CORNERS][0]
    src = np.array([tl, tr, br, bl])
    print("src ", src)

    # Keep upper-left corner in place & move the other three corners
    # relative to it.
    x_offset = (CONFIG_NUM_COLUMN_CORNERS-1)*CONFIG_SQUARE_LENGTH_INCHES*CONFIG_IMAGE_DPI
    y_offset = (CONFIG_NUM_ROW_CORNERS-1)*CONFIG_SQUARE_LENGTH_INCHES*CONFIG_IMAGE_DPI
    dst = np.array([
        tl,
        [tl[0] + x_offset, tl[1]],
        [tl[0] + x_offset, tl[1] + y_offset],
        [tl[0], tl[1] + y_offset]
    ])
    print("dst ", dst)
    M = cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return M, src, dst


def calibration_pipeline(img_path, save_path):
    filename = os.path.basename(img_path)
    full_output_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    full_output_path = full_output_path_no_extension + "_calibration.txt"
    # if os.path.exists(full_output_path):
    #     raise ValueError(f"{full_output_path} already exists; "
    #         "aborting to prevent overwriting it.")

    # Must be 8-bit for calibration.
    image = cv.imread(img_path, cv.IMREAD_COLOR)

    _, camera_mtx, dist, r_matrix, tvecs = calibrate(image)

    # As part of writing the calibration configs, we need to calculate
    # a perspective matrix.  To do that we need to undistort the target
    # image itself, find the outermost 4 corners, and run
    # getPerspectiveTransform vs the desired 4 corner positions.
    undistorted_image, _, _ = undistort(image, camera_mtx, dist)
    # Writing this file for debugging only.
    cv.imwrite(
        full_output_path + "-undist-nopersp.tif", undistorted_image,
        params=[cv.IMWRITE_TIFF_XDPI, CONFIG_IMAGE_DPI, cv.IMWRITE_TIFF_YDPI, CONFIG_IMAGE_DPI,
                cv.IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE])

    perspective_matrix, src, dst = find_perspective_with_cb_detection(undistorted_image)

    calibration = {
        "camera_matrix": camera_mtx.tolist(),
        "dist": dist.tolist(),
        "r_matrix": r_matrix.tolist(),
        "t_vecs": tvecs.tolist(),
        "perspective_matrix": perspective_matrix.tolist(),
        "perspective_src": src.tolist(),
        "perspective_dst": dst.tolist(),
    }
    print(calibration)
    with open(full_output_path, "w") as outfile:
        json.dump(calibration, outfile, indent=2)


def print_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        total_error += error
    print(f"mean error: {total_error/len(objpoints)}")


def read_offsets(offsets_file):
    offsets = {}
    with open(offsets_file, "r") as openfile:
        offsets = json.load(openfile)
    return offsets


def read_calibration(calibration_input_file):
    calibration = {}
    with open(calibration_input_file, "r") as openfile:
        calibration = json.load(openfile)
    # print(calibration)
    return calibration


def embed_in_target_area(img, offsets):
    # consider the entire scanner bed, with upper-left as (0,0)
    # "target" x,y,w,h defines a rectangle of interest for the distortion target
    # "roi" x,y defines a rectangle of interest for the scan area w/the card
    # (don't need roi w,h as that is just the image dimensions)

    target_x = offsets["target_x"]
    target_y = offsets["target_y"]
    target_w = offsets["target_w"]
    target_h = offsets["target_h"]
    roi_x = offsets["roi_x"]
    roi_y = offsets["roi_y"]
    roi_rotation = offsets["roi_rotation"]

    if roi_rotation == "cw":
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif roi_rotation == "ccw":
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif roi_rotation == "180":
        img = cv.rotate(img, cv.ROTATE_180)

    big_img = np.zeros((target_h, target_w, 3), img.dtype)
    x_offset = roi_x - target_x
    y_offset = roi_y - target_y
    (img_h, img_w, _) = np.shape(img)
    big_img[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img

    # cv.imshow("big_img", big_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return big_img


def crop_with_offsets(img, original_w, original_h, offsets):
    target_x = offsets["target_x"]
    target_y = offsets["target_y"]
    target_w = offsets["target_w"]
    target_h = offsets["target_h"]
    roi_x = offsets["roi_x"]
    roi_y = offsets["roi_y"]
    roi_rotation = offsets["roi_rotation"]

    x_offset = roi_x - target_x
    y_offset = roi_y - target_y
    if roi_rotation == "cw" or roi_rotation == "ccw":
        tmp = original_h
        original_h = original_w
        original_w = tmp
    # print(y_offset, y_offset+original_h, x_offset, x_offset+original_w)
    cropped_img = img[y_offset:y_offset+original_h, x_offset:x_offset+original_w]

    # cv.imshow("cropped_img", cropped_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Now need to reverse the rotation we did at the beginning.
    if roi_rotation == "cw":
        cropped_img = cv.rotate(cropped_img, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif roi_rotation == "ccw":
        cropped_img = cv.rotate(cropped_img, cv.ROTATE_90_CLOCKWISE)
    elif roi_rotation == "180":
        cropped_img = cv.rotate(cropped_img, cv.ROTATE_180)

    # cv.imshow("cropped_img", cropped_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return cropped_img


# https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
def get_scaled_camera_matrix(camera_mtx, scaling_factor):
    return np.array([camera_mtx[0] * scaling_factor, camera_mtx[1] * scaling_factor, camera_mtx[2]])


def camera_cal(calibration_input_file, linear_image, offsets_file):
    (original_height, original_width, _) = np.shape(linear_image)
    offsets = {}
    if offsets_file:
        offsets = read_offsets(offsets_file)
        linear_image = embed_in_target_area(linear_image, offsets)
    calibration = read_calibration(calibration_input_file)
    # r_matrix and tvecs are only needed to calculate re-projection
    # error, so no need to grab those.
    camera_mtx = np.array(calibration["camera_matrix"])
    dist = np.array(calibration["dist"])
    perspective_matrix = np.array(calibration["perspective_matrix"])

    linear_dst, _, _ = undistort(linear_image, camera_mtx, dist)
    (height, width, _) = np.shape(linear_dst)
    # The full magic_hand combines this warp with the affine warp that follows it, which is technically better (one
    # fewer linear interpolation). This is to test just the camera_calibrate piece.
    linear_dst = cv.warpPerspective(linear_dst, perspective_matrix, (width, height), flags=cv.INTER_LINEAR)
    if offsets:
        linear_dst = crop_with_offsets(linear_dst, original_width, original_height, offsets)
    return linear_dst


def undistort_pipeline(path, calibration_input_file, save_path, offsets_file=None):
    # 16-bit for rendering
    image_16 = cv.imread(path, cv.IMREAD_UNCHANGED)
    is_16bit = isinstance(image_16[0][0][0], np.uint16)
    scalar = 65535 if is_16bit else 255
    linear_image = cv.pow(image_16/scalar, PROPHOTO_GAMMA)

    linear_dst = camera_cal(calibration_input_file, linear_image, offsets_file)

    filename = os.path.basename(path)
    full_save_path_no_extension = os.path.join(save_path, filename).split(".tif")[0]
    full_output_path = full_save_path_no_extension + "-undist.tif"
    if is_16bit:
        dst = np.uint16(scalar * cv.pow(linear_dst, 1/PROPHOTO_GAMMA))
    else:
        dst = np.uint8(scalar * cv.pow(linear_dst, 1/PROPHOTO_GAMMA))
    cv.imwrite(
        full_output_path, dst,
        params=[cv.IMWRITE_TIFF_XDPI, CONFIG_IMAGE_DPI, cv.IMWRITE_TIFF_YDPI, CONFIG_IMAGE_DPI])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    parser.add_argument(
        "-c", dest="calibration_output_folder", type=str,
        help="full calibration output file path. the folder must exist but the file should not."
        " (cannot use with -o)")
    parser.add_argument(
        "-f", dest="offsets_file", type=str,
        help="file that describes offsets of the distortion target and image ROIs. for use only with -c")
    parser.add_argument(
        "-i", dest="calibration_input_file", type=str,
        help="input file that describes a calibration. for use only with -c")
    parser.add_argument(
        "-o", dest="output_folder", type=str,
        help="output folder for undistorted images. (cannot use with -c)")
    args = parser.parse_args()
    if ((args.output_folder and args.calibration_output_folder)
        or (not args.output_folder and not args.calibration_output_folder)):
        raise ValueError("Must use exactly one of -c or -o")
    if args.output_folder:
        if not os.path.isdir(args.output_folder):
            raise ValueError("Output path is not a folder")
        if not os.path.exists(args.output_folder):
            raise ValueError("Output path must already exist on the filesystem")
        if not os.path.exists(args.calibration_input_file):
            raise ValueError(
                f"Input calibration file {args.calibration_input_file} does not exist")
        paths = []
        for i in args.path:
            path = glob.glob(i)
            if isinstance(path, list) and path:
                paths.extend(path)
        if not paths:
            raise ValueError("No valid paths were provided")
        for i in paths:
            i = os.path.abspath(i)
            if os.path.exists(i):
                undistort_pipeline(
                    i,
                    args.calibration_input_file,
                    args.output_folder,
                    args.offsets_file)
            else:
                raise ValueError("{} does not exist".format(i))
    elif args.calibration_output_folder:
        if len(args.path) > 1:
            raise ValueError(
                "Multiple input calibration images are not supported (TODO for future"
                " version)")
        for i in args.path:
            i = os.path.abspath(i)
            if os.path.exists(i):
                calibration_pipeline(i, args.calibration_output_folder)
            else:
                raise ValueError("{} does not exist".format(i))
