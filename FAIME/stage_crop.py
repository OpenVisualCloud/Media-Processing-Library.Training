# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from stage import Stage, StageEnum
import logging
import settings
import database
import copy
import multiprocessing as mp
import os
from os import path
import utils
from shutil import copyfile
from PIL import Image

STAGE_NAME = 'StageCrop'

class StageCrop(Stage):

    def __init__(self):
        self.enum = StageEnum.CROP
        pass

    def crop(self, set, scene):

        if 'hr_path' not in scene:
            logging.warning("WARNING: HR folder for scene does not exist")
            return

        HRfoldername = scene['hr_path']
        if not path.exists(HRfoldername):
            logging.warning("WARNING: HR folder for scene does not exist:" + HRfoldername)
            return

        Cropfoldername = path.join(HRfoldername, "Crop")

        docrop = True
        if not path.exists(Cropfoldername):
            os.makedirs(Cropfoldername)
        else:
            files = utils.GetFiles(Cropfoldername)
            if len(files) > 0:
                logging.warning("WARNING..Crop folder for scene exists and contains frames: " + Cropfoldername)
                if not set.crop_overwrite:
                    docrop = False

        if docrop:
            logging.info(
                "Cropping " + HRfoldername + " to " + Cropfoldername + "  crop_num_frames=" + str(set.crop_num_frames) +
                "  crop fraction="+str(set.crop_fraction) +
                "  crop_divisible="+str(set.crop_divisible))
            files = utils.GetFiles(HRfoldername)
            numfiles = len(files)
            if set.crop_num_frames == -1:
                firstframe = 0
                lastframe = numfiles -1
            else:
                firstframe = int(max(0, numfiles/2 - set.crop_num_frames/2))
                lastframe = min(numfiles-1, firstframe + set.crop_num_frames-1)

            for i in range(firstframe, lastframe+1):
                basename = os.path.basename(files[i])
                cropfilename = os.path.join(Cropfoldername, basename)

                im = Image.open(files[i])
                width, height = im.size
                croppedwidth = width
                croppedheight = height
                if set.crop_fraction != 1.0:
                    croppedwidth = int(max(0, min(width, width*set.crop_fraction)))
                    croppedheight = int(max(0, min(height, height*set.crop_fraction)))
                    x = int((width - croppedwidth)/2)
                    y = int((height - croppedheight)/2)
                    im = im.crop((x,y, x+croppedwidth, y+croppedheight))

                if set.crop_divisible != 1:
                    croppedwidth = croppedwidth - croppedwidth % set.crop_divisible
                    croppedheight = croppedheight - croppedheight % set.crop_divisible
                    im = im.crop((0, 0, croppedwidth, croppedheight))
                im.save(cropfilename)

            scene['cropped'] = True
            scene['cropped_path'] = Cropfoldername
            scene['cropped_num_frames'] = numfiles
            scene['cropped_frame_width'] = croppedwidth
            scene['cropped_frame_height'] = croppedheight
        pass

    def ExecuteStage(self):
        logging.info("StageCrop..executing stage")

        if settings.set.crop_skip_stage:
            logging.info("StageCrop..skipping stage")
            return

        if 	settings.set.crop_num_frames == -1 and settings.set.crop_fraction == 1.0 and settings.set.crop_divisible ==1:
            logging.info("StageCrop..settings are default values..skipping stage")
            return

        # get a list of scene indices
        sceneindices = database.db.getSceneIndices()

        if settings.set.multiprocess:
            processes = []

            logging.info("StageCrop..starting multiprocess..# of scenes = " + str(len(sceneindices)))

            # In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
            # Python interpreter is started for each new multiprocessing.Process.This means that all global variables
            # are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
            # the spawned processes.
            # https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
            setcopy = copy.deepcopy(settings.set)

            scenecopies = []
            for sceneid in sceneindices:
                scenecopy = copy.deepcopy( database.db.getScene(sceneid))
                scenecopies.append(scenecopy)
                p = mp.Process(target=self.crop, args=(setcopy, scenecopy))
                processes.append(p)

            [p.start() for p in processes]
            [p.join() for p in processes]

            # copy back
            settings.set = copy.deepcopy(setcopy)
            for index, sceneid in enumerate(sceneindices):

                scene = database.db.getScene(sceneid)
                scene['cropped'] = scenecopies[index]['cropped']
                scene['cropped_path'] = scenecopies[index]['cropped_path']
                scene['cropped_frame_width'] = scenecopies[index]['cropped_frame_width']
                scene['cropped_frame_height'] = scenecopies[index]['cropped_frame_height']
                scene['cropped_num_frames'] = scenecopies[index]['cropped_num_frames']


        else:
            for sceneid in sceneindices:
                scene = database.db.getScene(sceneid)
                self.crop(settings.set, scene)


        database.db.save()
        database.db.list()
        logging.info("StageCrop..complete for all scenes")