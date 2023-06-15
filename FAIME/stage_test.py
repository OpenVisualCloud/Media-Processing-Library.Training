# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import coloredlogs
from stage import Stage, StageEnum
import settings
import database
import os
import os.path
from os import path
import warnings
warnings.filterwarnings("ignore")
import copy
import multiprocessing as mp
import glob
import shutil
from timeit import default_timer as timer
import json
from PIL import Image
import utils

STAGE_NAME = 'StageTest'

class StageTest(Stage):

    def __init__(self):
        self.enum = StageEnum.TEST
        pass

    def TestRaisrFFmpeg(self, set, foldername, lr_folder, test_folder):
        import ffmpeg
        parameters = {}
        filter_path = set.test_RaisrFF_filter_path
        parameters['ratio'] = set.test_upscale_factor
        parameters['bits'] = set.test_RaisrFF_bits
        parameters['range'] = set.test_RaisrFF_range
        parameters['threadcount'] = set.test_RaisrFF_threadcount
        parameters['filterfolder'] = filter_path
        parameters['blending'] = set.test_RaisrFF_blending
        parameters['passes'] = set.test_RaisrFF_passes
        parameters['mode'] = set.test_RaisrFF_mode

        scenetestfolder = os.path.join(test_folder, foldername)
        numfiles = len(utils.GetFiles(scenetestfolder))
        if not set.test_overwrite:
            if os.path.exists(scenetestfolder):
                  if numfiles > 0:
                    logging.info("WARNING..Folder exists, skipping testing: " + scenetestfolder)
                    return False

        curPath = os.path.abspath(os.getcwd())
        # filter_base_path = os.path.join(curPath, "raisr-native-library")
        if set.test_RaisrFF_filter_path != "":
            if not os.path.exists(set.test_RaisrFF_filter_path):
                logging.info("WARNING..Raisr FFmpeg filter path does not exist, skipping testing: " +
                             set.test_RaisrFF_filter_path)
                return False
        # Get paths for the lr and hr images
        lrfiles = utils.GetFiles(lr_folder)
        parameters_str = ''
        for key,value in parameters.items():
            parameters_str+=(':').join([key,str(value)])+','
        logging.info("INFO...Current ffmpeg parameters: " + parameters_str[:-1])
        for index, file in enumerate(lrfiles):
            ext = os.path.splitext(file)[1]
            if index % 20 == 0:
                logging.info('INFO...upscaling {} {} of {}'.format(foldername, index, len(lrfiles)))
            try:
                (
                    ffmpeg
                    .input(file)
                    .filter('raisr', **parameters)
                    .output(os.path.join(scenetestfolder,'%04d%s'%(index,ext)))
                    .run(quiet=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print(e.stderr)
                return False
        return True

    def TestInterpolation(self, set, foldername, lr_folder, test_folder, algorithm):
        import cv2
        import numpy as np
        if algorithm != "nearest_neighbor" and algorithm != "bilinear" and algorithm != "bicubic" and algorithm != "area" and algorithm != "lanczos":
            logging.info("WARNING..unsuported interpolation algorithm: " + algorithm)
            return False

        scenetestfolder = os.path.join(test_folder, foldername)
        if os.path.exists(scenetestfolder):
            if not set.test_overwrite:
                numfiles = len(utils.GetFiles(scenetestfolder))
                if numfiles > 0:
                    logging.info("WARNING..Folder exists, skipping testing: " + scenetestfolder)
                    return False
        else:
            os.mkdir(scenetestfolder)

        lrfiles = utils.GetFiles(lr_folder)

        for LRvideofilename in lrfiles:

            LRim: np.ndarray = cv2.imread(LRvideofilename)

            newwidth = LRim.shape[1] * set.test_upscale_factor
            newheight= LRim.shape[0] * set.test_upscale_factor

            if algorithm == "nearest_neighbor":
                scaledimage = cv2.resize(LRim,
                                  dsize=(newwidth,newheight),
                                  interpolation=cv2.INTER_NEAREST)
            elif algorithm == "bilinear":
                scaledimage = cv2.resize(LRim,
                                  dsize=(newwidth,newheight),
                                  interpolation=cv2.INTER_LINEAR)
            elif algorithm == "area":
                scaledimage = cv2.resize(LRim,
                                  dsize=(newwidth,newheight),
                                  interpolation=cv2.INTER_AREA)
            elif algorithm == "bicubic":
                scaledimage = cv2.resize(LRim,
                                  dsize=(newwidth,newheight),
                                  interpolation=cv2.INTER_CUBIC)
            elif algorithm == "lanczos":
                scaledimage = cv2.resize(LRim,
                                  dsize=(newwidth,newheight),
                                  interpolation=cv2.INTER_LANCZOS4)
            head_tail = os.path.split(LRvideofilename)
            test_path = os.path.join(scenetestfolder, head_tail[1])
            cv2.imwrite(test_path, scaledimage)
        return True

    def WriteTimeAverage(self, scenetestfolder, start, end):
        numfiles = len(utils.GetFiles(scenetestfolder))
        if numfiles > 0:
            secondsperimage = (end - start)/numfiles
            timefilepath = os.path.join(scenetestfolder,"timeaverage.txt")
            timefile = open(timefilepath, "w")
            timefile.write(str(secondsperimage))
            timefile.close()

    def SaveImage(self, image, height, width, path, overlap):
        img = Image.new('RGB', (width + 2 * overlap, height + 2 * overlap), 255)
        img.paste(image)
        # save the file, overwriting if necessary
        img.save(path)

    def Crop(self, image, height, width, overlap, numdivisions):
        for i in range(numdivisions):
            for j in range(numdivisions):
                box = (j * width - overlap, i * height - overlap, (j + 1) * width + overlap, (i + 1) * height + overlap)
                yield image.crop(box)

    def CreateTiles(self, set, lr_folder):
        # first check if resolution can be divided equally
        lrfiles = utils.GetFiles(lr_folder)

        im = Image.open(lrfiles[0])
        imgwidth, imgheight = im.size
        if imgwidth % set.test_tile_division != 0 or imgheight  % set.test_tile_division != 0:
            logging.warning("WARNING: lr frame size (" + str(imgwidth) + "," + str(imgheight) + ") is not a multiple of the tile division: " + str(set.test_tile_division))
            return False
        cropwidth = int(imgwidth / set.test_tile_division)
        cropheight = int(imgheight / set.test_tile_division)

        for lrfile in lrfiles:
            basename = os.path.split(lrfile)[1].rsplit('.', 1)[0]

            for i, piece in enumerate(self.Crop(im, cropheight, cropwidth, set.test_tile_overlap, set.test_tile_division)):
                tile_name = "tile" + str(i).zfill(2)
                tile_folder_name = os.path.join(lr_folder, tile_name)
                if not os.path.exists(tile_folder_name):
                    os.mkdir(tile_folder_name)
                tile_path = os.path.join(lr_folder, tile_name, basename + ".png")

                self.SaveImage(piece, cropheight, cropwidth, tile_path, set.test_tile_overlap)
        return True

    def JoinTiles(self, set, algorithm, lr_folder, scenetestfolder):

        # Join the tile test images
        # Get a list of the lr files for the scene
        lrfiles = utils.GetFiles(lr_folder)

        # copy the tile parameters
        offset = set.test_tile_overlap
        division = set.test_tile_division
        scaling = set.test_upscale_factor

        # imageList is a list of tile images for each lr file that will need to be stitched together
        imageList = []

        # determine the number of tests that we should run (1 per tile)
        numtests = set.test_tile_division * set.test_tile_division

        # for each lr file
        for lrfile in lrfiles:

            # get the base name of the file (without the extension)
            basename = os.path.split(lrfile)[1].rsplit('.', 1)[0]

            # for each tile
            for i in range(numtests):
                # get the file path for the tile
                tile_name = "tile" + str(i).zfill(2)
                test_tile_folder_name = os.path.join(scenetestfolder, tile_name)
                test_tile_test_file_path = os.path.join(test_tile_folder_name, basename + ".png")
                if not os.path.exists(test_tile_test_file_path):
                    logging.warning(
                        "WARNING: Tile file not found: " + test_tile_test_file_path + " Unable to complete testing")
                    return
                # append the file path to the list
                imageList.append(test_tile_test_file_path)

            # Open the first tile to get it's size
            image = Image.open(imageList[0])
            tilewidth, tileheight = image.size

            # determine the total size of the stitched image
            totalwidth = (tilewidth - 2 * offset * scaling) * division
            totalheight = (tileheight - 2 * offset * scaling) * division

            # create a new image which will be pasted with sections of the tiles
            new_im = Image.new('RGB', (totalwidth, totalheight))

            # start at the top
            y_offset = 0
            # go through each row
            for im in range(0, len(imageList), set.test_tile_division):
                # go through each column
                x_offset = 0
                for i in range(set.test_tile_division):
                    # open the corresponding tile image
                    image1 = Image.open(imageList[im + i])
                    # define a box to crop the image to the section we will paste
                    box = (
                    offset * scaling, offset * scaling, tilewidth - offset * scaling, tileheight - offset * scaling)
                    # do the crop
                    imagec = image1.crop(box)
                    # do the paste
                    new_im.paste(imagec, (x_offset, y_offset))
                    # go to the next column
                    x_offset += tilewidth - scaling * offset * 2
                y_offset += tileheight - scaling * offset * 2

            # all done pasting.  Determine the test image name and save
            testfilepath = os.path.join(scenetestfolder, basename + ".png")
            new_im.save(testfilepath)

    def RunTestAlgorithm(self, algorithm, set, foldername, lr_folder, test_folder, hr_folder):

        logging.info("StageTest..Algorithm=" + algorithm + " foldername=" + foldername)
        ret = False
        if algorithm == 'raisr_ffmpeg':
            ret = self.TestRaisrFFmpeg(set, foldername, lr_folder, test_folder)
        elif algorithm == "nearest_neighbor":
            ret = self.TestInterpolation(set, foldername, lr_folder, test_folder, "nearest_neighbor")
        elif algorithm == "bilinear":
            ret = self.TestInterpolation(set, foldername, lr_folder, test_folder, "bilinear")
        elif algorithm == "area":
            ret = self.TestInterpolation(set, foldername, lr_folder, test_folder, "area")
        elif algorithm == "bicubic":
            ret = self.TestInterpolation(set, foldername, lr_folder, test_folder, "bicubic")
        elif algorithm == "lanczos":
            ret = self.TestInterpolation(set, foldername, lr_folder, test_folder, "lanczos")
        else:
            logging.warning("WARNING: unsupported algorithm: " + algorithm)
            ret = False

        return ret

    def CreateDiffs(self, set, scenetestfolder, hr_folder, difference_style, difference_multiplier):
        import numpy as np
        import cv2

        test_paths = utils.GetFiles(scenetestfolder)
        hr_paths = utils.GetFiles(hr_folder)

        diff_path = os.path.join(scenetestfolder, "diffs")
        if not os.path.exists(diff_path):
            os.mkdir(diff_path)

        for index, test_path in enumerate(test_paths):
            test_data_bgr = cv2.imread(test_path)
            hr_data_bgr = cv2.imread(hr_paths[index])

            # create the difference data
            if difference_style == 'RGB':
                # RGB
                test_data_rgb= cv2.cvtColor(test_data_bgr, cv2.COLOR_BGR2RGB)
                hr_data_rgb = cv2.cvtColor(hr_data_bgr, cv2.COLOR_BGR2RGB)

                diff_data = np.abs(test_data_bgr - hr_data_bgr)
                diff_data = diff_data.astype('uint8')
            else:
                # https://docs.opencv.org/4.5.2/d3/d50/group__imgproc__colormap.html
                test_data_yuv = cv2.cvtColor(test_data_bgr, cv2.COLOR_BGR2YUV)
                test_data_yuv = test_data_yuv.astype('int32')
                hr_data_yuv = cv2.cvtColor(hr_data_bgr, cv2.COLOR_BGR2YUV)
                hr_data_yuv = hr_data_yuv.astype('int32')

                if test_data_yuv.shape != hr_data_yuv.shape:
                    logging.info("WARNING: Unable to create difference file for "+test_path+ " HR and Test image sizes don''t match")
                    diff_data = np.clip(0, 255, difference_multiplier * np.abs(test_data_yuv - test_data_yuv))
                else:
                    diff_data = np.clip(0,255,difference_multiplier*np.abs(test_data_yuv - hr_data_yuv))
                diff_data[:, :, 1] = diff_data[:, :, 0]
                diff_data[:, :, 2] = diff_data[:, :, 0]
                diff_data = diff_data.astype('uint8')
                colormap_keys = [
                    "AUTUMN",            "BONE",                "JET",               "WINTER",            "RAINBOW",
                    "OCEAN",             "SUMMER",              "SPRING",            "COOL",              "HSV",
                    "PINK",              "HOT",                 "PARULA",            "MAGMA",             "INFERNO",
                    "PLASMA",            "VIRIDIS",             "CIVIDIS",           "TWILIGHT",          "TWILIGHT_SHIFTED",
                    "TURBO",             "DEEP_GREEN"]
                colormap_values = [
                    cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE,    cv2.COLORMAP_JET,     cv2.COLORMAP_WINTER,   cv2.COLORMAP_RAINBOW,
                    cv2.COLORMAP_OCEAN,  cv2.COLORMAP_SUMMER,  cv2.COLORMAP_SPRING,  cv2.COLORMAP_COOL,     cv2.COLORMAP_HSV,
                    cv2.COLORMAP_PINK,   cv2.COLORMAP_HOT,     cv2.COLORMAP_PARULA,  cv2.COLORMAP_MAGMA,    cv2.COLORMAP_INFERNO,
                    cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED,
                    cv2.COLORMAP_TURBO,  cv2.COLORMAP_DEEPGREEN]
                colormap_dict = {colormap_keys[i]: colormap_values[i] for i in range(len(colormap_keys))}
                if difference_style in colormap_dict:
                    diff_data = cv2.applyColorMap(diff_data, colormap_dict[difference_style])
                    diff_data = cv2.cvtColor(diff_data, cv2.COLOR_BGR2RGB)
                else :
                    logging.warning("WARNING: unsupported difference style: " + difference_style)
                    return

            diff_image = Image.fromarray(diff_data)

            # save the difference file
            head, tail = os.path.split(test_path)
            diff_png_path = os.path.join(diff_path, tail)
            diff_image.save(diff_png_path)

        if set.test_video_create:
            utils.CreateVideo(diff_path)


    def Test(self, set, scene, lr_folder, test_folder, hr_folder):
        import cv2

        # Configure Logging
        logger = logging.getLogger(__name__)
        coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

        # start timing for the scene's test
        start = timer()

        # Create the scene test folder
        foldername = scene['folder_name']
        scenetestfolder = os.path.join(test_folder, foldername)
        if not os.path.exists(scenetestfolder):
            os.mkdir(scenetestfolder)

        # remove time file
        timefilepath = os.path.join(scenetestfolder, "timeaverage.txt")
        if os.path.exists(timefilepath):
            os.remove(timefilepath)

        # set upscale ratio
        if set.test_upscale_factor == 0:
            hrfiles = utils.GetFiles(hr_folder)
            lrfiles = utils.GetFiles(lr_folder)
            hr_img = cv2.imread(hrfiles[0])
            lr_img = cv2.imread(lrfiles[0])
            scalefactor = int(hr_img.shape[0]/lr_img.shape[0])
            if (scalefactor == 2) or (scalefactor == 4):
                set.test_upscale_factor = scalefactor
                logging.info("StageTest..test_upscale_factor calculated=" + str(set.test_upscale_factor))
            else:
                logging.warning("WARNING: unsupported upscale factor calculated=" + str(scalefactor) + " from HR:"+hrfiles[0] + " and LR:"+lrfiles[0])
                return
        ret = True
        if set.test_tile:
            # Doing tiles

            # Create the tiles for all of the images in the lr folder for this scene
            if not self.CreateTiles(set, lr_folder):
                logging.warning("WARNING: unable to create tiles for scene=" + str(scene["scene_index"]) + " " + scene["folder_name"])

            # determine the number of tests that we should run (1 per tile)
            numtests = set.test_tile_division*set.test_tile_division

            # for each test
            for i in range(numtests):
                # determine the name for the tile (e.g. tile00, tile01, tile02,...)
                tile_name = "tile" + str(i).zfill(2)

                # determine the lr folder for the tile images.  This should have been created in CreateTiles
                tile_lr_folder_name = os.path.join(lr_folder, tile_name)
                if not os.path.exists(tile_lr_folder_name):
                    logging.warning("WARNING: Tile lr folder not found: " + tile_lr_folder_name + " Unable to complete testing")
                    return

                # make sure tile test folder exists
                tile_test_folder = os.path.join(test_folder, scenetestfolder, tile_name)
                if not os.path.exists(tile_test_folder):
                    os.mkdir(tile_test_folder)

                # Run the test for this tile
                ret = self.RunTestAlgorithm(set.test_algorithm.lower(), set, tile_name, tile_lr_folder_name, scenetestfolder, hr_folder)

            if ret:
                self.JoinTiles(set, set.test_algorithm.lower(),lr_folder, scenetestfolder)

        else:
            ret = self.RunTestAlgorithm(set.test_algorithm.lower(), set, foldername, lr_folder, test_folder, hr_folder)

        # finish timing
        end = timer()
        if ret:
            self.WriteTimeAverage(scenetestfolder, start, end)
            if set.test_video_create:
                utils.CreateVideo(scenetestfolder)
            if set.test_difference_style.upper() != "NONE" and set.test_difference_style.upper() != "":
                self.CreateDiffs(set, scenetestfolder, hr_folder, set.test_difference_style.upper(), set.test_difference_multiplier)
        logging.info("StageTest.. scene=" + str(foldername) + " finished")
        return ret

    def ExecuteStage(self):
        logging.info("StageTest..executing stage")

        if settings.set.test_skip_stage:
            logging.info("StageTest..skipping stage")
            return

        # test folder
        if not os.path.exists(settings.set.absolute_test_folder):
            os.mkdir(settings.set.absolute_test_folder)
        test_folder = os.path.join(settings.set.absolute_test_folder, settings.set.test_name)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)

        # Make sure LR folder exists
        if not os.path.exists(settings.set.absolute_LR_folder):
            logging.warning("StageTest..Unable to test.  LR folder does not exist: "+ settings.set.absolute_LR_folder)
            return

        # clear tests
        if settings.set.test_clear_tests:
            logging.info("StageTest..clearing tests")

            # remove test files.  Use ignore_errors in the probable case where the folder contains files
            sub_folders_pathname = os.path.join(test_folder,"*")
            sub_folders_list = glob.glob(sub_folders_pathname)
            for sub_folder in sub_folders_list:
                shutil.rmtree(sub_folder, ignore_errors=True)

            # also clear tests in the database
            for scene in database.db.scenes:
                scene['tests'] = {}

        if settings.set.test_algorithm == "":
            logging.info("StageTest..no test algorithm specified")
            return

        sceneindices = database.db.getSceneIndices()
        if len(sceneindices) == 0:
            logging.warning("StageTest..WARNING - no scenes to process")
            return

        if settings.set.multiprocess:
            processes = []

            logging.info("StageTest.starting multiprocess..# of scenes = " + str(len(sceneindices)))

            # In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
            # Python interpreter is started for each new multiprocessing.Process.This means that all global variables
            # are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
            # the spawned processes.
            # https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
            setcopy = copy.deepcopy(settings.set)
            scene_folders = []

            # find the maximum scene to process
            lastsceneindex = sceneindices[-1]
            logging.info("StageTest...last scene = " + str(lastsceneindex))
            for sceneindex in sceneindices:

                logging.info("StageTest...processing scene " + str(sceneindex) )
                scene = database.db.getScene(sceneindex)

                lr_folder = os.path.join(settings.set.absolute_LR_folder, scene['folder_name'])
                if not os.path.exists(lr_folder):
                    logging.warning("StageTest..WARNING - low res folder does not exist for scene: " + lr_folder)
                    continue

                # hr folder
                if scene['cropped']:
                    hr_folder = scene['cropped_path']
                else:
                    hr_folder = scene['hr_path']
                if settings.set.test_video_create:
                    utils.CreateVideo(hr_folder)

                p = mp.Process(target=self.Test, args=(setcopy, scene, lr_folder, test_folder, hr_folder))
                processes.append(p)

                p.start()
                scene_folders.append(scene["folder_name"])
                if (len(processes) == settings.set.max_num_processes):
                    # we reached the maximum number of processes.  Wait until 1 finishes
                    q = processes.pop(0)
                    q.join()

                if sceneindex == lastsceneindex:
                    # Last scene is in processes.  Finish all joins
                    folderstr = ' '.join(map(str, scene_folders))
                    logging.info("StageTest..finishing scenes: " + folderstr)
                    [p.join() for p in processes]
                    processes.clear()
                    scene_folders.clear()

            # copy back
            settings.set = copy.deepcopy(setcopy)

        else:
            for sceneindex in sceneindices:

                logging.info("StageTest.starting processing scene " + str(sceneindex) + " of " + str(len(sceneindices)))
                scene = database.db.getScene(sceneindex)

                # if individual scenes are specified
                if len(settings.set.individual_scenes) > 0 and sceneindex not in settings.set.individual_scenes:
                    logging.info("StageTest...skipping scene " + str(sceneindex) + " (not in individual_scenes list)")
                    continue

                lr_folder = os.path.join(settings.set.absolute_LR_folder, scene['folder_name'])
                if not os.path.exists(lr_folder):
                    logging.warning("StageTest..WARNING - low res folder does not exist for scene: " + lr_folder)
                    continue

                # hr folder
                if scene['cropped']:
                    hr_folder = scene['cropped_path']
                else:
                    hr_folder = scene['hr_path']
                if settings.set.test_video_create:
                    utils.CreateVideo(hr_folder)

                logging.info("StageTest...processing scene " + str(sceneindex) )

                self.Test(settings.set, scene, lr_folder, test_folder, hr_folder)
                database.db.save()

        # read the average test time for each scene
        logging.info("StageTest..reading average test time")
        for sceneindex in sceneindices:
            scene = database.db.getScene(sceneindex)

            # update the test_path
            scene['test_path'] = os.path.join(test_folder, scene['folder_name'])

            # create the time file
            timefilepath = os.path.join(scene['test_path'], "timeaverage.txt")
            if not os.path.exists(timefilepath):
                logging.warning("StageTest...scene average test time file not found: " + timefilepath)
                continue
            timefile = open(timefilepath, "r")
            timestr = timefile.read()
            timefile.close()

            # Create dictionary for the test in the database
            if settings.set.test_name not in scene['tests']:
                scene['tests'][settings.set.test_name] = {}

            if scene['cropped']:
                hr_folder = scene['cropped_path']
            else:
                hr_folder = scene['hr_path']

            # add several settings for this test run
            scene['tests'][settings.set.test_name]['average_test_time'] = float(timestr)
            scene['tests'][settings.set.test_name]['test_path'] = os.path.join(test_folder, scene['folder_name'])
            scene['tests'][settings.set.test_name]['lr_path'] = os.path.join(settings.set.absolute_LR_folder, scene['folder_name'])
            scene['tests'][settings.set.test_name]['hr_path'] = hr_folder

            # save the metrics file
            foldername = scene['folder_name']
            metricsfile = os.path.join(test_folder, foldername, 'metrics.json')
            with open(metricsfile, 'w') as f:
                json.dump(scene['tests'][settings.set.test_name], f, indent=2, separators=(',', ':'))

            logging.info("StageTest.." + scene['folder_name'] + " average test time per frame (seconds): " + timestr)

        # save the database
        database.db.save()
        logging.info("StageTest..complete for all scenes")
