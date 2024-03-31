import argparse
import glob
import os


def rename(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            ext = file.split(".")[-1]
            tokens = file.split("-")
            [set_abbrev, number] = tokens[0].split("_")
            new_file = set_abbrev + "_en_" + number + "-" + tokens[1] + "." + ext
            new_file_full = os.path.join(path, new_file)
            # print(new_file_full)
            original_file_full = os.path.join(path, file)
            # print(original_file_full)
            os.rename(original_file_full, new_file_full)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="rename files")
    parser.add_argument(
        "path",
        metavar="input_image",
        type=str,
        nargs="+",
        help="path or list of paths to an image.")
    # parser.add_argument(
    #     "-o",
    #     dest="output_folder",
    #     type=str,
    #     help="output folder for centered images",
    #     required=True)
    args = parser.parse_args()

    # if not os.path.isdir(args.output_folder):
    #     raise ValueError("Output path is not a folder")
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
            rename(i)
        else:
            raise ValueError("{} does not exist".format(i))
