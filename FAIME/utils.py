# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import video
from pathlib import Path
from typing import List, Union

def convert_images_to_video(files: List[str],
                            output_file: Path,
                            fps: float = 30):
    """
    Convert a list of image files to a video file.

    :param files: A list of paths to the image files
    :param output_file: The path to the output video file
    :param fps: The frames per second of the output video
    """
    import av
    from PIL import Image
    from numpy import asarray

    # Get the codec based on the file extension
    codec = {'.mp4': 'h264', '.yuv': 'rawvideo'}.get(output_file.suffix)
    if codec is None:
        raise Exception(f'Unsupported video format: {output_file.suffix}')

    # Create the output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the stream
    stream = None
    try:
        # Open the output video file
        with av.open(str(output_file), 'w') as container:
            # Loop over the image files
            for file in files:
                # Open the image file
                img = Image.open(file)

                # Convert the image to a numpy array
                data = asarray(img)

                # If the stream hasn't been initialized, create it
                if stream is None:
                    stream = container.add_stream(codec, rate=fps)
                    stream.width = data.shape[1]
                    stream.height = data.shape[0]

                    # Set the bitrate for H.264 codec
                    if codec == 'h264':
                        # Kush Gauge values
                        # https: // originaltrilogy.com / topic / Restoration - tips - Kush - Gaugetm / id / 16250
                        _MOTION_RANK = 4  # High motion
                        _BITRATE_K = 0.07  # H.264
                        stream.bit_rate = (stream.width * stream.height * fps *
                                           _MOTION_RANK * _BITRATE_K)

                # Create a video frame from the image
                frame = av.VideoFrame.from_image(img)

                # Encode the frame and write it to the video file
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush the stream
            for packet in stream.encode():
                container.mux(packet)

    except BaseException:
        # If an error occurs, delete the output file
        if output_file.is_file():
            output_file.unlink()
        raise


def CreateVideo(folder: Union[str, Path], overwrite: bool = True):
    """
    Create a video from the image files in the specified directory.

    :param folder: The path to the directory containing the images
    :param overwrite: Whether to overwrite an existing video file with the same name (defaults to True)
    """

    # Get the image files in the specified directory
    files = GetFiles(folder)

    # Get the path to the output video file
    videopath = os.path.join(folder, 'frames.mp4')

    # Check if the video file already exists
    if overwrite or not os.path.exists(videopath):
        # If it doesn't exist or if overwriting is enabled, create the video
        convert_images_to_video(files, Path(videopath))


def RemoveDir(directory: Union[str, Path]) -> None:
    """
    Recursively delete all files and subdirectories within the specified directory.

    :param directory: The path to the directory to delete
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            RemoveDir(item)
        else:
            item.unlink()
    directory.rmdir()


def RemoveFiles(directory: Union[str, Path]) -> None:
    """
    Remove all files in the specified directory, including files in subdirectories.

    :param directory: The directory to remove the files from.
    :type directory: Union[str, Path]
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            RemoveDir(item)
        else:
            item.unlink()


def get_size(bytes: int, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format and return a string representation of the size.

    :param bytes: The number of bytes to scale
    :param suffix: The suffix to append to the scaled value (defaults to "B" for bytes)
    :return: A string representation of the scaled size, in the format X.XXUNIT
    """

    # Define the scaling factor and units
    factor = 1024
    units = ["", "K", "M", "G", "T", "P"]

    # Loop through the units and divide the bytes by the factor until the value is less than the factor
    for unit in units:
        if bytes < factor:
            # Return the formatted size string
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor



def GetFiles(directory: str, recursive: bool = False) -> List[str]:
    """
    This function returns a list of all files with the specified extensions in the given directory.
    If the recursive flag is set to True, it will also include files in subdirectories of the given directory.
    The file extensions to include in the list are defined in the 'types' tuple.
    The list of files is sorted in alphabetical order.
    """
    # Define the file types to include in the list
    types = ('*.png', '*.jpg', '*.y4m')

    # Initialize an empty list to hold the files
    inputfiles = []

    # Loop over the file types
    for files in types:
        # Add all files with the current type to the list
        inputfiles.extend(glob.glob(os.path.join(directory, files)))

    # If the recursive flag is True, get files in subdirectories
    if recursive:
        # Loop over the subdirectories in the current directory
        for subdir in os.listdir(directory):
            # Get the full path to the subdirectory
            d = os.path.join(directory, subdir)
            # If the path is a directory, get its files
            if os.path.isdir(d):
                # Add the files in the subdirectory to the list
                inputfiles.extend(GetFiles(d, recursive))
    # Sort the files in alphabetical order
    inputfiles.sort()
    # Return the list of files
    return inputfiles


def ReadVideo(video_path):
    import cv2
    video_capture = cv2.VideoCapture(video_path)
    if video.Video.getBits == video.TEN_BIT:
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',''))
        video_capture.set(cv2.CAP_PROP_FORMAT, cv2.CV_16U)
        video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return video_capture
