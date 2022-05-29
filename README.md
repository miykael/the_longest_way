# The Longest Way by Christoph Rehage (https://thelongestway.com/)

This repository contains two scripts, `align_images.py` and `create_video.py`:

- To run `align_images.py`, you need to provide an input folder containing all images that you want to align. Each face in an image will be stored in an output folder, assuming that the face covers at least a certain size of the image.
- To run `create_video.py`, you need to provide an input folder containing multiple images (ideally aligned) to create a video, similar to the one shown in [Noah's 7777 video](https://www.youtube.com/watch?v=DC1KHAxE7mo) (see [here](https://github.com/miykael/noah_ages) to corresponding repo).

### Setup

To execute this script, you might need to install a few additional python packages. Assuming that you already have a python environment on your system, you can simply run the following commands in your terminal and you should be good to go!

```bash
# Install all required python packages
pip install opencv-python numpy scipy tqdm dlib scikit-image
```

### Align images

Once these packages are installed, you can run the script directly via the terminal.

```bash
# Run with the -h flag to see all your options
python align_images.py -h

# Execute script with default parameters (using images in the 'raw' folder)
python align_images.py -i raw
```

Here's an overview of all additional parameters:

```txt
  -i INPUT, --input INPUT
                        Path to the folder containing the raw images.
                        (default: raw)
  -o OUTPUT, --output OUTPUT
                        Name of the output folder where the aligned images
                        should be stored in (default: processed)
  -l DLIB, --dlib DLIB  Path to dlib folder containing
                        'shape_predictor_5_face_landmarks.dat' and
                        'mmod_human_face_detector.dat' file (default: dlib)
  -f, --force           If true, already processed images will be overwritten.
                        Otherwise already processed images are skipped.
                        (default: False)
  -c CHIP_SIZE [CHIP_SIZE ...], --chip_size CHIP_SIZE [CHIP_SIZE ...]
                        Image resolution of the output image. Choose '4240
                        2832' for 12MP camera resolution, or '3840 2160' for
                        4k resolution. (default: [3840, 2160])
  -u UPSAMPLE, --upsample UPSAMPLE
                        Factor to upsample image. Can be 0, 1, 2, ... This
                        parameter improves alignment but increases computation
                        time. Default 1 seems to be a good compromise.
                        (default: 1)
  -t OUTTYPE, --outtype OUTTYPE
                        File type of output image. Can be 'jpg', 'png', 'tif',
                        ... (default: jpg)
  -d DETECTOR, --detector DETECTOR
                        Specify if hog or cnn detector should be used. CNN is
                        more advanced and better, but takes longer. (default:
                        hog)
  -p PADDING, --padding PADDING
                        Padding around the extracted face chip. To only
                        extract the face, set to 0. At 1, the face chip will
                        be padded by the width of a face chip. For this
                        project, it is recommend to set it somewhere between 3
                        and 4. (default: 3.25)
```

As mentioned in the text, `-u` (upsampling) can make the output more accurate, but also much slower. Should be fine with default parameters. `-d cnn` (using the cnn detector) might slightly improve results, but also here, computation time is much longer. `-p` (padding) can be used to see more or less of the image (try it out with `-p 1` or `-p 6` to see what I mean).


### Create video

To create a video out of some images, please ensure that you have ffmpeg installed.

```bash
# If you want to create a video à la Noah's 7777 project, you might need to also install ffmpeg first
brew install ffmpeg  # only works for mac users
```

Once you have ffmpeg on your system, simply use the `create_video.py` script to create videos similar to the one shown in [Noah's 7777 video](https://www.youtube.com/watch?v=DC1KHAxE7mo).

```bash
# Run with the -h flag to see all your options
python create_video.py -h

# Execute script with default parameters (using images in the 'processed' folder)
python create_video.py -i processed
```

Here's an overview of all additional parameters:

```txt
  -i INPUT, --input INPUT
                        Path to the folder containing the images you want to combine in
                        a video. (default: processed)
  -o OUTPUT, --output OUTPUT
                        Name of output file. (default: video_compilation)
  -s SMOOTH, --smooth SMOOTH
                        Number of images that should be averaged per time point. 7
                        images means each frame is an average of a week. (default: 7)
  -t STEP_SIZE, --step_size STEP_SIZE
                        Step size of images that should be used as center of the
                        smoothing. I.e. if set to 2, then every 2nd image will be
                        skipped (and only appears in the average, but not as center
                        image). (default: 1)
  -f FPS, --fps FPS     Frames per second in the video. 30 frames mean 1-month per
                        second. (default: 30)
  -r RESOLUTION [RESOLUTION ...], --resolution RESOLUTION [RESOLUTION ...]
                        Image resolution of the video. Up to you, but could be '3840
                        2160' (4k), or '1920 1080' (Full HD). (default: [3840, 2160])
```
