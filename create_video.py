import argparse
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm
from skimage.transform import resize


def main(args):

    # Collect filenames of images to combine in video
    filenames = sorted(args.input.glob("*?.*"))

    # Extract number of images
    N_total = len(filenames)

    # Get start indeces for images
    ids = [idx * args.step_size for idx in range((N_total + args.smooth) // args.step_size + 1)]

    print(f"Total video length is {len(ids)/args.fps:.2f} seconds.")

    # Establish name of output file and temporary folder
    tmp_img_idx = Path(f"imgs_for_video_{args.output}")

    # Remove already existing folder
    [t.unlink() for t in tmp_img_idx.glob('*?.*')]
    tmp_img_idx.rmdir()

    # Create temporary folder to store images into
    tmp_img_idx.mkdir(parents=True, exist_ok=True)

    # To keep track of what was already loaded
    already_loaded = []

    # Loop through all images
    for i in tqdm(ids):

        # Collect indeces of images
        imgs_idx = np.arange(np.clip(i - args.smooth, 0, N_total - 1), np.clip(i, 0, N_total - 1) + 1)

        # Collect images relevant for the group
        group_names = np.array(filenames)[imgs_idx]

        # Detect which one is new to load
        new_to_load = np.setdiff1d(group_names, already_loaded)

        if len(new_to_load) == 0:
            pass
        elif i == 0:
            imgs_group = np.array([io.imread(f) for f in new_to_load])
        else:
            img_new = np.array([io.imread(f) for f in new_to_load])
            imgs_group = np.vstack((imgs_group, img_new))

        # Cut imgs_group to write size
        n_offset = i - N_total
        if n_offset <= 0:
            n_offset = 0
        elif n_offset % 2 == 0:
            n_offset -= 1
        imgs_group = imgs_group[-args.smooth + n_offset :]

        # Create composition image
        img_comp = np.mean(imgs_group, axis=0).astype("int")

        # Create out_filename
        out_filename = tmp_img_idx / f"{(i + 1):05d}.jpg"

        # Resize images to desired resolution
        image_resized = resize(img_comp, args.resolution, anti_aliasing=True, preserve_range=True)

        # Save composition image
        io.imsave(out_filename, image_resized.astype("uint8"))

        # Keep track of what has already been loaded
        already_loaded = group_names

    import subprocess

    cmd = ["ffmpeg", "-y", "-r", str(args.fps), "-vcodec", "mjpeg", "-i", str(tmp_img_idx) + "/%05d.jpg", "-vcodec", "libx264", str(args.output) + ".mp4"]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.call(cmd)


if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser(
        description="Using ffmpeg, this program creates a video of multiple images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        default="processed",
        help="Path to the folder containing the images you want to combine in a video.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        default="video_compilation",
        help="Name of output file.",
    )
    parser.add_argument(
        "-s",
        "--smooth",
        type=int,
        required=False,
        default=7,
        help="""Number of images that should be averaged per time point. 7 images means each frame is an average
        of a week.""",
    )
    parser.add_argument(
        "-t",
        "--step_size",
        type=int,
        required=False,
        default=1,
        help="""Step size of images that should be used as center of the smoothing. I.e. if set to 2,
        then every 2nd image will be skipped (and only appears in the average, but not as center image).""",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        required=False,
        default=30,
        help="Frames per second in the video. 30 frames mean 1-month per second.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        required=False,
        default=[3840, 2160],
        help="Image resolution of the video. Up to you, but could be '3840 2160' (4k), or '1920 1080' (Full HD).",
    )
    args = parser.parse_args()

    # Convert resolution list to integers
    args.resolution = [int(r) for r in args.resolution[::-1]]

    # Check if output resolution has two dimensions
    if len(args.resolution) != 2:
        print(f"Output resolution {args.resolution} not supported!")
        exit()

    # Run main script
    main(args)
