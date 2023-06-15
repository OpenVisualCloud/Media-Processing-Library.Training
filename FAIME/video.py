# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


from typing import IO, Dict, Iterator, List, Optional, Tuple, Union
import settings
import ffmpeg
import errno
import logging
import coloredlogs
import numpy as np

TEN_BIT_VIDEO_FORMATS = ['yuv420p10le', 'yuv422p10le', 'yuv444p10le',
                         'yuv420p10be', 'yuv422p10be', 'yuv444p10be']
TEN_BIT_IMAGE_FORMATS = ['rgb48be','rgb48le']
TEN_BIT_FORMATS = TEN_BIT_VIDEO_FORMATS + TEN_BIT_IMAGE_FORMATS
TEN_BIT = '10'
EIGHT_BIT_VIDEO_FORMATS = ['yuv420p','yuv422p','yuv444p',
                           'yuvj420p','yuvj422p, yuvj444p']
EIGHT_BIT_IMAGE_FORMATS = ['rgb24']
EIGHT_BIT_FORMATS = EIGHT_BIT_VIDEO_FORMATS + EIGHT_BIT_IMAGE_FORMATS
EIGHT_BIT = '8'

class Video():


    class VideoCapture():
        """
        A class for capturing frames from a video file.
        """

        def __init__(self):
            """
            Initialize the capture object.
            """
            pass

        def read(self):
            """
            Read a single frame from the video file.

            :return: A tuple containing a boolean value indicating whether the frame was successfully read and the frame data.
            """
            if self.index >= self.scene_frames:
                # End of video reached
                return False, None
            else:
                # Increment the index and return the current frame
                self.index += 1
                return True, self.scene[self.index - 1]


    class VideoCaptureDisk(VideoCapture):
        """
        A class for reading a video stored on disk as a series of .png frames.
        """

        def __init__(self, video_path: str):
            """
            Initialize the video reader.

            :param video_path: The path to the directory containing the video frames.
            """
            import cv2, re
            self.cv2 = cv2
            self.re = re
            self.scene = self._ReadScene(video_path)
            self.scene_frames, _, _, _ = self.scene.shape
            self.index = 0

        def _ReadImage(self, image_path: str):
            """
            Read a single image from disk.

            :param image_path: The path to the image file.
            :return: The image data as a NumPy array.
            """
            im = self.cv2.imread(
                image_path, self.cv2.IMREAD_ANYDEPTH | self.cv2.IMREAD_COLOR
            )
            return im

        def _ReadScene(self, scene_path: str):
            """
            Read all of the frames in the video from disk.

            :param scene_path: The path to the directory containing the video frames.
            :return: The video data as a NumPy array.
            """
            import glob, os
            scene = []
            image_pattern = os.path.join(scene_path, "*.png")
            for file in sorted(glob.glob(image_pattern), key=self._image_number):
                image = self._ReadImage(file)
                scene.append(image)
            scene = np.asarray(scene)
            return scene

        def _image_number(self, image: str):
            """
            Extract the index of the frame from the filename.

            :param image: The filename of the image.
            :return: The index of the frame as an integer.
            """
            frameindex_re = self.re.compile("([0-9]{4})")
            result = self.re.search(frameindex_re, image)
            if result:
                return int(result.group(1))
            else:
                return -1


    class VideoCaptureMemory(VideoCapture):
        """
        VideoCaptureMemory is a class that extends the VideoCapture class by storing the video frames in memory
        instead of on disk.
        """

        def __init__(self, frames):
            """
            Initializes the VideoCaptureMemory object with a list of frames representing the video.

            :param frames: A list of numpy arrays representing the frames of the video.
            """
            # Call the parent class's constructor
            super().__init__()

            # Store the frames in the object's attribute
            self.frames = frames

            # Calculate the number of frames in the video
            self.num_frames = len(frames)

            # Set the current frame index to 0
            self.index = 0

    def __init__(self, video_path, scene_name):
        """Initialize a video object with the path to the video and the scene_name. scene_name used for
        logging purposes.

        Args:
            video_path (str): path to the video file
            scene_name (str): scene_name for underlying FAIME implementation
        """
        self.video_path = video_path
        self.vc = None
        self.temp_dir = None
        self.scene_name = scene_name
        self._GetMetadata()

    def read(self):
        """
        Reads the next frame in the video.

        :return: A tuple containing a boolean indicating if there are more frames left in the video and the
                next frame of the video (as a numpy array).
        """
        # Check if there are more frames left in the video
        if self.index >= self.num_frames:
            # Return False and None if there are no more frames
            return False, None
        else:
            # Increment the current frame index
            self.index += 1

            # Return True and the current frame
            return True, self.frames[self.index - 1]


    @staticmethod
    def isImage(media_path: str, ignore_errors: bool = False) -> bool:
        """
        Determine whether the given media path is an image file by checking if its codec is 'mjpeg' or 'png'.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            bool: True if the file is an image, False otherwise.
        """
        try:
            return ffmpeg.probe(media_path)['streams'][0]['codec_name'] in ('mjpeg','png')
        except ffmpeg.Error as e:
            handleFFmpegError(e, media_path)
            if not ignore_errors:
                raise e

    @staticmethod
    def isVideo(media_path: str, ignore_errors: bool = False) -> bool:
        """
        Determine whether the given media path is a video file by checking if its codec is not 'mjpeg' or 'png'.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            bool: True if the file is a video, False otherwise.
        """
        try:
            return not Video.isImage(media_path)
        except ffmpeg.Error as e:
            handleFFmpegError(e, media_path)
            if not ignore_errors:
                raise e

    @staticmethod
    def getBits(media_path: str, ignore_errors: bool = False) -> int:
        """
        Determine the number of bits used by the given media file.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            int: The number of bits used by the file, either 8 or 10.
        """
        try:
            pix_fmt = ffmpeg.probe(media_path)['streams'][0]['pix_fmt']
            if pix_fmt in TEN_BIT_FORMATS:
                return TEN_BIT
            elif pix_fmt in EIGHT_BIT_FORMATS:
                return EIGHT_BIT
        except ffmpeg.Error as e:
            handleFFmpegError(e,media_path)
            if not ignore_errors:
                raise e


    @staticmethod
    def getResolution(media_path: str, ignore_errors: bool = False) -> str:
        """
        Get the resolution of the given media file.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            str: The resolution of the file, in the format 'width x height'.
        """
        try:
            probe = ffmpeg.probe(media_path)
            return 'x'.join(map(str, [probe['streams'][0]['width'], probe['streams'][0]['height']]))
        except ffmpeg.Error as e:
            handleFFmpegError(e,media_path)
            if not ignore_errors:
                raise e

    @staticmethod
    def getPixFmt(media_path: str, ignore_errors: bool = False) -> str:
        """
        Get the pixel format of the given media file.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            str: The pixel format of the file.
        """
        try:
            return ffmpeg.probe(media_path)['streams'][0]['pix_fmt']
        except ffmpeg.Error as e:
            handleFFmpegError(e,media_path)
            if not ignore_errors:
                raise e

    @staticmethod
    def getCodec(media_path: str, ignore_errors: bool = False) -> str:
        """
        Get the codec of the given media file.

        Args:
            media_path (str): The path to the media file.
            ignore_errors (bool): Whether to ignore any errors that occur while checking the file.

        Returns:
            str: The codec of the file.
        """
        try:
            return ffmpeg.probe(media_path)['streams'][0]['codec_name']
        except ffmpeg.Error as e:
            handleFFmpegError(e,media_path)
            if not ignore_errors:
                raise e

    @staticmethod
    def is10bitStatic(video_path: str, ignore_errors: bool = False) -> bool:
        """
        Determine whether the video is 10-bit.

        Args:
            video_path (str): The path to the video file.

        Returns:
            bool: True if the video is 10-bit, False otherwise.
        """
        try:
            probe = ffmpeg.probe(video_path)
            pix_fmt = next(stream["pix_fmt"] for stream in probe["streams"] if stream["codec_type"] == "video")
            return pix_fmt in TEN_BIT_FORMATS
        except ffmpeg.Error as e:
            handleFFmpegError(e,video_path)
            if not ignore_errors:
                raise e


    def _GetMetadata(self) -> None:
        """
        Extract metadata from the video file and store it in `self.probe`.
        """
        self.probe = ffmpeg.probe(self.video_path)

    @property
    def width(self) -> int:
        """
        Get the width of the video.

        Returns:
            int: The width of the video.
        """
        return self.probe['streams'][0]['width']

    @property
    def height(self) -> int:
        """
        Get the height of the video.

        Returns:
            int: The height of the video.
        """
        return self.probe['streams'][0]['height']

    @property
    def pix_fmt(self) -> str:
        """
        Get the pixel format of the video.

        Returns:
            str: The pixel format of the video.
        """
        return self.probe['streams'][0]['pix_fmt']

    @property
    def format(self) -> str:
        """
        Get the format of the video.

        Returns:
            str: The format of the video.
        """
        return self.probe['format']['format_name']

    @property
    def is10bit(self) -> bool:
        """
        Determine whether the video is 10-bit.

        Returns:
            bool: True if the video is 10-bit, False otherwise.
        """
        return self.pix_fmt in TEN_BIT_FORMATS


    @property
    def frames(self) -> int:
        """
        Get the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        if 'nb_frames' in self.probe['streams'][0]:
            # The number of frames is stored directly in the metadata
            return int(self.probe['streams'][0]['nb_frames'])
        else:
            # Calculate the number of frames from the duration and frame rate
            duration = self.getProperty('duration')
            frame_rate = self.getProperty('r_frame_rate')
            if not duration or not frame_rate:
                # If the duration or frame rate is not available, return None
                return None

            # Split the frame rate into its numerator and denominator
            frame_rate = frame_rate.split('/')
            import math
            if frame_rate[1] == '1':
                # If the denominator is 1, simply multiply the numerator by the duration
                return math.floor(int(frame_rate[0]) * float(duration))
            else:
                # If the denominator is not 1, divide the numerator by the denominator and multiply by the duration
                return math.floor(int(frame_rate[0]) // int(frame_rate[1]) * float(duration))


    @property
    def fps(self) -> int:
        """
        Get the frames per second of the video.

        Returns:
            int: The frames per second of the video.
        """
        return round(float(self.frames) / float(self.getProperty('duration')))

    @property
    def frame_size(self) -> int:
        """
        Get the size of each frame in the video.

        Returns:
            int: The size of each frame in the video, in bytes.
        """
        return self.width * self.height * 3

    def getProperty(self, property: str) -> Optional[Union[int, str]]:
        """
        Get the specified property of the video.

        Args:
            property (str): The name of the property to get.

        Returns:
            Union[int, str]: The value of the specified property, or None if the property does not exist.
        """
        if property in self.probe['streams'][0]:
            return self.probe['streams'][0][property]
        if property in self.probe['format']:
            return self.probe['format'][property]
        return None

    def setTempDir(self, temp_dir: str) -> None:
        """
        Set the temporary directory to use for storing intermediate files.

        Args:
            temp_dir (str): The path to the temporary directory.
        """
        import os
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        self.temp_dir = temp_dir

    def isOpenable(self) -> bool:
        """
        Determine whether the video can be opened.

        Returns:
            bool: True if the video can be opened, False otherwise.
        """
        import cv2
        vc = cv2.VideoCapture(self.video_path)
        openable = vc.isOpened()
        cv2.destroyAllWindows()
        return openable


    def isOpen(self) -> bool:
        """
        Determine whether the video is currently open.

        Returns:
            bool: True if the video is open, False otherwise.
        """
        if self.vc:
            return True
        else:
            return False

    def open(self) -> bool:
        """
        Open the video.

        Returns:
            bool: True if the video was successfully opened, False otherwise.
        """
        import cv2, os
        if not self.is10bit:
            # If the video is not 10-bit, use the built-in OpenCV method
            self.vc = cv2.VideoCapture(self.video_path)
        else:
            # If the video is 10-bit, use the custom _OpenSciKitMethod()
            self._OpenSciKitMethod()
        if self.vc:
            return True
        else:
            return False


    def _OpenDiskMethod(self) -> None:
        """
        Open the video using the "disk" method.
        """
        start_frame = 0
        end_frame = int(self.frames) - 1
        if not self.temp_dir:
            # If no temporary directory is set, use the default "temp" directory
            self.setTempDir('temp')
        output_folder = self.temp_dir
        # Split the video into individual frames using FFmpeg
        self._FFmpegSplit(start_frame, end_frame, output_folder)
        # Open the video using the custom VideoCaptureDisk() method
        self.vc = self.VideoCaptureDisk(output_folder)


    def _ConvertVideo(self) -> Optional[str]:
        """
        Convert the video to the yuv4mpegpipe format with a YUV420p10le pixel format.

        Returns:
            Optional[str]: The path to the converted video, or None if an error occurred.
        """
        import ffmpeg
        import os
        if not self.temp_dir:
            # If no temporary directory is set, use the default "temp" directory
            self.setTempDir('temp')
        self.temp_vid_path = os.path.join(self.temp_dir, 'temp_vid.y4m')
        try:
            # Use FFmpeg to convert the video
            (ffmpeg
            .input(self.video_path)
            .output(self.temp_vid_path, **{'f':'yuv4mpegpipe','pix_fmt':'yuv420p10le','strict':-1})
            .run(quiet=True, overwrite_output=True))
            return self.temp_vid_path
        except ffmpeg.Error as e:
            import shutil
            # Delete the temporary directory if an error occurred
            shutil.rmtree(self.temp_dir)
            print(e.stderr)
            return


    def _EachFrame(self, stream: IO) -> Iterator[str]:
        """
        Iterate over the frames in a video stream.

        Args:
            stream (IO): The video stream.

        Yields:
            Iterator[str]: The individual frames in the video stream.
        """
        buffer = ''  # Initialize an empty buffer
        while True:
            # Read a chunk of the video stream
            chunk = stream.read(self.frame_size)
            if not chunk:  # If there are no more chunks, return the final frame
                yield buffer
                break
            buffer += chunk  # Add the chunk to the buffer
            while True:
                try:
                    # Split the buffer on the "FRAME" string to extract the current frame
                    part, buffer = buffer.split('FRAME\n',1)
                except ValueError:
                    break  # If "FRAME" is not in the buffer, move on to the next chunk
                else:
                    yield part  # If "FRAME" is in the buffer, yield the current frame



    def _OpenY4MStream(self):
        """
        Open a video stream using the y4m format.

        This function converts the video to the yuv4mpegpipe format if it is not
        already in that format. It then reads the video file frame by frame and
        decodes each frame using the Y4M.Reader class. The decoded frames are
        added to a list and the list is used to create a new VideoCaptureList object
        that can be used to access the frames of the video.
        """
        from inspect import _void
        import math
        import multiprocessing as mp
        from Raisr import Y4M

        video_path: str = self.video_path
        if self.format != 'yuv4mpegpipe':
            # If the video is not in the yuv4mpegpipe format, convert it
            video_path = self._ConvertVideo()
        frames = []
        with open(video_path, 'rb') as fh:
            # Read the header of the video file
            header = b''
            while b'FRAME\n' not in header:
                header = header + fh.read(1)
            reader = Y4M.Reader(_void)
            reader.set_headers(header)
            # Iterate over each frame in the video
            for frame in self._EachFrame(fh):
                # Decode the frame and add it to the list of frames
                y4 = reader.decode(frame)
                frames.append(np.reshape(np.asarray(list(y4.y) + list(y4.uv),dtype=np.uint16), (self.height, self.width, 3)))

        self.vc = self.VideoCaptureList(frames)



    def _OpenY4MMethod(self):
        """Open a video file in Y4M format using the multiprocessing module.

        This method uses the `multiprocessing` module to split the video file into
        multiple processes and then decode each frame in parallel. The resulting
        frames are stored in a list which is then passed to the `VideoCaptureList`
        class to create a video capture object.
        """
        import math
        import multiprocessing as mp
        import psutil

        video_path = self.video_path
        if self.format != 'yuv4mpegpipe':
            video_path = self._ConvertVideo()

        # Determine the maximum number of processes to use. This should be the
        # number of logical CPU cores multiplied by two, or the total number of
        # frames in the video (whichever is smaller).
        MAX_PROCCESES = min(int(psutil.cpu_count(logical=True) * 2), self.frames)

        # Split the video file into multiple processes.
        processes = []
        frames = None
        headers = None
        with open(video_path, 'rb') as fh:
            frames = fh.read()
            if not frames:
                logging.error('{}.. Unable to read video file frames'.format(self.scene_name))
                exit()
            frames = frames.split(b'FRAME\n')
            headers = frames.pop(0)
        frames_per_process = math.ceil(len(frames) / MAX_PROCCESES)
        manager = mp.Manager()

        # Use a list to store the decoded frames.
        frame_list = manager.list([0] * len(frames))
        for i in range(MAX_PROCCESES):
            p = mp.Process(target=self._GetNumpyArray, args=(
                frames[i * frames_per_process:i * frames_per_process + frames_per_process],
                i * frames_per_process,
                headers,
                frame_list))
            processes.append(p)

        # Start the processes and wait for them to complete.
        [p.start() for p in processes]
        [p.join() for p in processes]
        self.vc = self.VideoCaptureList(frame_list)

    def _OpenSciKitMethod(self):
        """
        Open a video stream using the skvideo library.
        This method is used when the video is in a 10-bit format, as the cv2 library does not support 10-bit videos.
        """
        # Import the necessary libraries
        from skvideo import io

        frames = []  # A list to hold the frames of the video
        # Use the FFmpegReader class from the skvideo library to read the video
        reader = io.FFmpegReader(self.video_path, outputdict={'-pix_fmt':'yuv444p16le'})
        # Iterate over each frame in the video
        for frame in reader.nextFrame():
            # Add the frame to the list of frames
            frames.append(frame)
        # Set the video capture object to a VideoCaptureList object initialized with the list of frames
        self.vc = self.VideoCaptureList(frames)


    def _GetNumpyArray(self, frames: List[bytes], start_index: int, headers: bytes, output_list):
        """
        Convert the given frames to a list of numpy arrays and store them in the output list.

        This function uses the Y4M library to decode the frames and convert them to numpy arrays.
        The frames are inserted into the output list starting at the given start index.

        Args:
            frames (List[bytes]): The list of frames to convert.
            start_index (int): The index in the output list where the first frame will be inserted.
            headers (bytes): The header of the video file.
            output_list (List[np.ndarray]): The list where the converted frames will be stored.
        """
        from inspect import _void
        from Raisr import Y4M

        # Create a Y4M reader
        reader = Y4M.Reader(_void)
        reader.set_headers(headers)

        # Iterate over each frame in the list
        for i, f in enumerate(frames):
            # Decode the frame and add it to the output list at the correct index
            frame = reader.decode(f)
            output_list[i + start_index] = np.reshape(
                np.asarray(list(frame.y) + list(frame.uv), dtype=np.uint16),
                (self.height, self.width, 3))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video.

        Returns:
            A tuple containing a boolean value indicating if the read was successful and the frame as a NumPy array.
        """
        if not self.isOpen():
            # If the video is not open, return False and None
            return False, None
        else:
            # If the video is open, read the next frame using the underlying video capture object
            return self.vc.read()



    def close(self) -> bool:
        """Close the video stream.

        This method will delete the video capture object and destroy any open windows. If the video is not in the yuv4mpegpipe format,
        the temporary directory will be deleted.
        """
        if not self.isOpen():
            return True

        # Delete the video capture object
        del self.vc
        self.vc = None

        # Destroy any open windows
        if not self.is10bit:
            import cv2
            cv2.destroyAllWindows()
            return True

        # If the video is not in the yuv4mpegpipe format, delete the temporary directory
        if self.format != 'yuv4mpegpipe':
            import shutil
            try:
                if self.temp_dir:
                    shutil.rmtree(self.temp_dir)
            except OSError as e:
                # Log a warning if the temporary directory could not be removed
                logging.warning('{}.. Unable to remove temporary directory {}. Caught error {}'.format(self.scene_name, self.temp_dir, e.strerror))

        return True

    def set(self, key: int, val: int) -> bool:
        """Set the specified property for the video capture.

        This method sets the specified property for the video capture. If the video
        is not open or is 10-bit, the method returns False. Otherwise, it returns
        the result of calling the `set` method on the video capture object.

        Args:
            key: The property to set. This should be one of the constants defined
                in the cv2 module (e.g. cv2.CAP_PROP_FRAME_WIDTH).
            val: The value to set the property to.

        Returns:
            A boolean indicating whether the property was successfully set.
        """
        if not self.isOpen():
            return False

        if not self.is10bit:
            return self.vc.set(key, val)
        else:
            return False



    def _FFmpegSplit(self, start_frame: int, end_frame: int, output_folder: str, pix_fmt: str = 'rgb48le') -> None:
        """
        Split the video from start_frame to end_frame into frames of format specified by pix_fmt.

            Parameters
            ----------
            start_frame : int
                The start frame of the range of frames to be split.
            end_frame : int
                The end frame of the range of frames to be split.
            output_folder : string
                Path to the directory where the frames should be stored.
            pix_fmt : str, optional
                The pixel format of the frames to be split, by default 'rgb48le'

            Raises
            ------
            Exception
                If the output directory is not a directory.

            Returns
            -------
            None

        """
        import os
        if not os.path.isdir(output_folder):
            raise Exception('Output directory provided for FFmpeg Split is not a directory')
        (ffmpeg
        .input(self.video_path)
        .filter('select','between(n,{},{})'.format(start_frame, end_frame))
        .output("%s%%04d.png"%(os.path.join(output_folder,'')), **{'qscale:v':3, 'pix_fmt':pix_fmt, 'vsync':0, 'start_number':0})
        .run(quiet=True))


    def _OpenCVSplit(self, start_frame: int, end_frame: int, output_folder: str, frame_width: int, frame_height: int, scenedetection_metrics: list = [], downscalefactor: float = 1.0, down_scale_algorithm: int = 1) -> Tuple[list, list]:
        """
        Split the video from start_frame to end_frame into frames of size (frame_width, frame_height).

            Parameters
            ----------
            start_frame : int
                The start frame of the range of frames to be split.
            end_frame : int
                The end frame of the range of frames to be split.
            output_folder : string
                Path to the directory where the frames should be stored.
            frame_width : int
                Width of the frames to be split.
            frame_height : int
                Height of the frames to be split.
            scenedetection_metrics : list, optional
                List of the scenedetection metrics, by default []
            downscalefactor : float, optional
                Factor by which the frames should be scaled down, by default 1
            down_scale_algorithm : int, optional
                The algorithm to be used for downscaling, by default 1

            Returns
            -------
            values : list
                List of the scenedetection metrics for the frames split.
            values2 : list
                List of the scenedetection metrics for the frames split.
            framenames : list
                List of the file paths of the frames split.
        """
        import cv2
        import os
        interpolationalg = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

        framenames = []
        values = []
        values2 = []

        video_capture = cv2.VideoCapture(self.video_path)
        video_capture.set(1, start_frame)
        first_scene_frame = start_frame

        for framenum in range(start_frame, end_frame + 1):
            ret, frame = video_capture.read()

            if downscalefactor != 1.0:
                frame = cv2.resize(frame,
                                dsize=(int(frame_width / downscalefactor),
                            int(frame_height, downscalefactor)),
                                interpolation=interpolationalg[down_scale_algorithm])

            # determinte frame index
            indexinscene = framenum - first_scene_frame
            framepath = os.path.join(output_folder, "%04d.png" % (indexinscene))
            try:
                cv2.imwrite(framepath, frame)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logging.error('ERROR - no space left on disk to save frame')
                    return
            # keep track of the frame names, and values
            framenames.append(framepath)
            if len(scenedetection_metrics) >= framenum:
                values.append(scenedetection_metrics[framenum][0])
                values2.append(scenedetection_metrics[framenum][1])
            else:
                values.append(0.0)
                values2.append(0.0)
        return values, values2

    def SplitScene(self, scene: Dict[str, Union[str, int, float]], output_folder: str,
                scenedetection_metrics: List[Tuple[float, float]], settings: settings) -> Tuple[List[float], List[float]]:
        """Splits a scene into individual frames.

        This function takes a scene dictionary, an output folder, a list of scene detection metrics,
        and a settings object. It uses FFmpeg or OpenCV to split the scene into individual frames
        and saves them to the output folder. It then returns two lists of values calculated from
        the scene detection metrics.

        Args:
            scene: A dictionary containing information about the scene to be split
            output_folder: The path of the folder where the frames should be saved
            scenedetection_metrics: A list of tuples containing scene detection metrics
            settings: A settings object containing settings for the scene splitting process

        Returns:
            Two lists of values calculated from the scene detection metrics
        """
        import glob, os, re

        # Get the start and end frames of the scene
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']

        # Initialize two lists to store values calculated from the scene detection metrics
        values = []
        values2 = []

        # Check if the video is 10-bit
        if self.is10bit:
            # Use FFmpeg to split the scene
            self._FFmpegSplit(start_frame, end_frame, output_folder)

            # Change the current working directory to the output folder
            oldpwd = os.getcwd()
            os.chdir(output_folder)

            # Compile a regular expression to match the frame index in the file name
            frameindex_re = re.compile('([0-9]{4})')

            # Iterate over the files in the output folder
            for file in glob.glob('*.png'):
                # Use the regular expression to match the frame index in the file name
                result = re.search(frameindex_re, file)

                # Initialize a variable to store the frame number
                framenum = float('inf')

                # Check if the regular expression matched the frame index
                if result:
                    # Calculate the frame number from the index
                    index = int(result.group(1))
                    framenum = start_frame + index

                # Check if the frame number is within the bounds of the scene detection metrics
                if len(scenedetection_metrics) >= framenum:
                    # Append the values from the scene detection metrics to the values lists
                    values.append(scenedetection_metrics[framenum][0])
                    values2.append(scenedetection_metrics[framenum][1])
                else:
                    # Append default values to the values lists
                    values.append(0.0)
                    values2.append(0.0)

            # Change the current working directory back to the original folder
            os.chdir(oldpwd)
        else:
            # Get the width and height of the frames in the scene
            width = scene['frame_width']
            height = scene['frame_height']
            values, values2 = self._OpenCVSplit(start_frame, end_frame, output_folder,
                                                width, height, scenedetection_metrics,
                                                settings.split_downscalefactor,
                                                settings.split_downscale_algorithm)
            return values, values2

def handleFFmpegError(error: ffmpeg.Error, media_path: str) -> None:
    """Handles FFmpeg errors.

    This function takes an error object and a media path and logs the error message.
    If the error message indicates that the file or directory does not exist, it logs a
    warning message. If the error message indicates that the file is a directory, it logs
    an error message.

    Args:
        error: An FFmpeg error
        media_path: The path of the media file or directory

    Returns:
        None
    """
    # Print the error message from stderr
    print(error.stderr)

    # Check if the error message indicates that the file or directory does not exist
    if b'No such file or directory' in error.stderr:
        # Log a warning message
        logging.warning('{} is not a file or directory'.format(media_path))

    # Check if the error message indicates that the file is a directory
    elif b'Is a directory' in error.stderr:
        # Log an error message
        logging.error('ERROR: {} is a directory'.format(media_path))
    elif b'Invalid data' in error.stderr:
        # Log an error message
        logging.error('ERROR: {} is invalid data'.format(media_path))