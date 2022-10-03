import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from skimage.util import img_as_float
from tqdm import tqdm

import dlib


def crop_img(img, dim=(1920, 1080)):
    """Face chip outputs are squared, this function crops them back to the desired
    width to height ratio resolution"""
    ratio = dim[1] / dim[0]
    offset = int((img.shape[1] - (img.shape[0] * ratio)) / 2)
    return img[offset : offset + dim[1], ...]


def get_distance_to_center(img, rectangle):

    """Computes the distance of the detecte face to the center of the image."""

    # Extract rectangle center
    center_idx = rectangle.center()
    center_idx = [center_idx.y, center_idx.x]

    # Compute center and width discrepancies
    discrepancy_center = np.linalg.norm(np.divide(img.shape[:2], 2) - center_idx)

    # Return product of discrepancies
    return np.abs(discrepancy_center)


def zoom(img, zoom_factor=1.33):
    """Zoom main image to create background image"""
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


def get_mask(input_image):
    """This function returns a mask of connected black regions of at least a specific size."""
    color_mean = input_image.mean(-1)
    label_im, _ = ndimage.label(color_mean)

    # Find black regions
    mask_labels = [idx for idx in range(10) if color_mean[label_im == idx].sum() == 0]
    black_mask = np.any([label_im == m for m in mask_labels], axis=0)
    return black_mask


def main(args):

    """
    args.input = Path('raw')
    args.output = Path('processed')
    args.dlib = Path('dlib')
    args.force = True
    args.chip_size = [4368, 2912]
    args.upsample = 1
    args.intype = 'tif'
    args.outtype = 'tif'
    args.detector = 'hog' # or cnn
    args.shape = 68
    args.padding = 3.25
    """

    # Collect filenames of images to process
    filenames = sorted(args.input.glob(f"*.{args.intype.lower()}"))

    # dlib predictor and detector for face recognition
    if args.shape == 5:
        shape_predictor = dlib.shape_predictor(str(args.dlib / "shape_predictor_5_face_landmarks.dat"))
    elif args.shape == 68:
        shape_predictor = dlib.shape_predictor(str(args.dlib / "shape_predictor_68_face_landmarks.dat"))
    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(str(args.dlib / "mmod_human_face_detector.dat"))

    # Go through all image and align them
    for f in tqdm(filenames):

        # Check if output files were already created
        extension = f".{args.outtype.lower()}"
        filename_stem = str(Path(f).stem) + extension
        out_filename = args.output / filename_stem
        if out_filename.exists() and not args.force:
            continue

        # Load image
        im = (img_as_float(io.imread(f)) * 255).astype('uint8')

        # Get information about image size
        w, h = im.shape[:2]
        offset = (h - w) // 2

        # Center image in a canvas
        canvas = np.zeros((h, h, 3)).astype("uint8")
        canvas[offset:-offset, ...] = im

        # Detect faces with hog or cnn detector (CNN is more advanced but takes longer)
        if args.detector.lower() == "hog":
            rectangles = np.array([x if isinstance(x, dlib.rectangle) else x.rect for x in hog_detector(canvas, args.upsample)])

            # If detection failed, run with cnn_detector anyhow
            if len(rectangles) == 0:
                rectangles = np.array([x if isinstance(x, dlib.rectangle) else x.rect for x in cnn_detector(canvas, np.clip(args.upsample - 1, 0, None, dtype="int"))])

        else:
            rectangles = np.array([x if isinstance(x, dlib.rectangle) else x.rect for x in cnn_detector(canvas, np.clip(args.upsample - 1, 0, None, dtype="int"))])

        # If detection still failed, do nothing and inform
        if len(rectangles) == 0:
            print(f"No face found in {f} - file skipped.")
            continue

        # Sort detected faces according to size and distance from center
        rectangle_info = np.array([get_distance_to_center(canvas, r) for r in rectangles])
        idx_sort = np.argsort(rectangle_info)

        # Only keep most central face
        idx_sort = idx_sort[:1]
        rectangles = rectangles[idx_sort]

        # Extract landmarks and face chips
        landmarks = [shape_predictor(canvas, r) for r in rectangles]
        face_chips = [dlib.get_face_chip(canvas, l, size=args.chip_size[0], padding=args.padding) for l in landmarks]
        points = np.array([np.array([[e.x, e.y] for e in l.parts()]) for l in landmarks])

        # Only keep main face (comment the following line to also get other faces)
        #face_chips = face_chips[:1]

        # Crop face chips to right ratio and save them
        for idx, face in enumerate(face_chips):

            # Create output image
            img_final = crop_img(face, dim=args.chip_size)

            # Compute zoomed background to cover up black borders
            background = zoom(img_final, zoom_factor=1.33)
            w, h = img_final.shape[:2]
            center = np.array(background.shape) / 2
            x = int(center[0] - w / 2)
            y = int(center[1] - h / 2)
            background = background[x : x + w, y : y + h, ...]

            # Extract black mask label
            black_mask = get_mask(img_final)

            # Add background to original image
            img_final[black_mask] = background[black_mask]

            # Save aligned image
            output_name = str(out_filename).replace(extension, f"_{idx:02d}{extension}")
            io.imsave(output_name, img_final)


if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser(
        description="Using dlib, this program aligns a person's face to the center of an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        default="raw",
        help="Path to the folder containing the raw images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default="processed",
        help="Name of the output folder where the aligned images should be stored in",
    )
    parser.add_argument(
        "-l",
        "--dlib",
        type=Path,
        required=False,
        default="dlib",
        help="Path to dlib folder containing 'shape_predictor_5_face_landmarks.dat' and 'mmod_human_face_detector.dat' file",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If true, already processed images will be overwritten. Otherwise already processed images are skipped.",
    )
    parser.add_argument(
        "-c",
        "--chip_size",
        nargs="+",
        required=False,
        default=[4368, 2912],
        help="Image resolution of the output image. Choose '4368 2912' for 12.7MP camera resolution, or '3840 2160' for 4k resolution.",
    )
    parser.add_argument(
        "-u",
        "--upsample",
        type=int,
        required=False,
        default=1,
        help="Factor to upsample image. Can be 0, 1, 2, ... This parameter improves alignment but increases computation time. Default 1 seems to be a good compromise.",
    )
    parser.add_argument(
        "-it",
        "--intype",
        type=str,
        required=False,
        default="tif",
        help="File type of output image. Can be 'jpg', 'png', 'tif', ...",
    )
    parser.add_argument(
        "-ot",
        "--outtype",
        type=str,
        required=False,
        default="tif",
        help="File type of output image. Can be 'jpg', 'png', 'tif', ...",
    )
    parser.add_argument(
        "-d",
        "--detector",
        type=str,
        required=False,
        default="hog",
        help="Specify if hog or cnn detector should be used. CNN is more advanced and better, but takes longer.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        required=False,
        default=5,
        help="Number of landmarks to use for alignment. Possible values are 5 or 68.",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=float,
        required=False,
        default=3.25,
        help="""Padding around the extracted face chip. To only extract the face, set to 0. At 1, the face chip will
        be padded by the width of a face chip. For this project, it is recommend to set it somewhere between 3 and 4.""",
    )
    args = parser.parse_args()

    # Convert chip size list to integers
    args.chip_size = [int(c) for c in args.chip_size]

    # Check if output resolution has two dimensions
    if len(args.chip_size) != 2:
        print(f"Chip size {args.chip_size} not supported!")
        exit()

    # Create output folder
    args.output.mkdir(parents=True, exist_ok=True)

    # Run main script
    main(args)
