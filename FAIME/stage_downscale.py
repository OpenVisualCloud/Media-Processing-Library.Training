# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import random
import json
import coloredlogs
from stage import Stage, StageEnum
import settings
import database
import os
import os.path
from os import path
import copy
import multiprocessing as mp
import glob
import subprocess
import math
import utils

def modcrop(im,modulo):
    shape = im.shape
    size0 = shape[0] - shape[0] % modulo
    size1 = shape[1] - shape[1] % modulo
    if len(im.shape) == 2:
        out = im[0:size0, 0:size1]
    else:
        out = im[0:size0, 0:size1, :]
    return out

STAGE_NAME = 'StageDownscale'

class StageDownscale(Stage):

	def __init__(self):
		self.enum = StageEnum.DOWNSCALE
		pass

	def downscale(self, set, scene):
		import cv2
		import numpy as np
		# Configure Logging
		logger = logging.getLogger(__name__)
		coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

		# INTER_NEAREST – a nearest-neighbor interpolation
		# INTER_LINEAR – a bilinear interpolation (used by default)
		# INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
		# INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
		interpolationalg = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

		downscale_algorithm = []

		# create a folder name for the scene
		foldername = scene['folder_name']

		if 'hr_path' not in scene:
			logging.warning("WARNING: HR folder for scene does not exist")
			return

		if scene['cropped']:
			HRfoldername = scene['cropped_path']
		else:
			HRfoldername = scene['hr_path']
		if not path.exists(HRfoldername):
			logging.warning("WARNING: HR folder for scene does not exist:" + HRfoldername)
			return

		LRfoldername = os.path.join(set.absolute_LR_folder, foldername)
		dodownscale= True

		if not path.exists(LRfoldername ):
			os.makedirs(LRfoldername )
		else:
			files = utils.GetFiles(LRfoldername)
			if len(files) > 0:
				logging.warning("WARNING..LR folder for scene exists and contains frames: " + LRfoldername)
				if not set.downscale_overwrite:
					dodownscale = False

		if dodownscale:
			logging.info("Downscaling " + HRfoldername + " to " + LRfoldername + " factor=" + str(set.downscale_scalefactor))
			files = utils.GetFiles(HRfoldername)

			# remove all files in LR folder:
			utils.RemoveFiles(LRfoldername)

			for index, HRvideofilename in enumerate(files):
				# Log progress
				if index % 20 == 0:
					logging.info("INFO...downscaling " + scene['folder_name'] + " " + str(index) + " of " + str(len(files)) )

				basename = os.path.split(HRvideofilename)[1].rsplit('.', 1)[0]

				if set.downscale_format.upper() == "PNG":
					extension = ".png"
				else:
					extension = ".jpg"

				LRvideofilename = os.path.join(LRfoldername, basename + extension)

				im: np.ndarray = cv2.imread(HRvideofilename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
				if im.any() == None:
					logging.warning("WARNING: Unable to read HR file:" + HRvideofilename)
					continue

				im = modcrop(im,set.downscale_scalefactor)
				height, width, channels = im.shape
				if scene.get('same_size'):
					if index == 0:
						# first image
						if width % set.downscale_scalefactor != 0:
							logging.warning("WARNING...downscaling " + scene['folder_name'] + " width="+str(width) + " is not a multiple of the downscale factor: " + str(set.downscale_scalefactor))
						if height % set.downscale_scalefactor != 0:
							logging.warning("WARNING...downscaling " + scene['folder_name'] + " height="+str(height) + " is not a multiple of the downscale factor: " + str(set.downscale_scalefactor))
				else:
					if width % set.downscale_scalefactor != 0:
						logging.warning("WARNING...downscaling " + scene['folder_name'] + " " + HRvideofilename + " width=" + str(width) + " is not a multiple of the downscale factor: " + str(set.downscale_scalefactor))
					if height % set.downscale_scalefactor != 0:
						logging.warning("WARNING...downscaling " + scene['folder_name'] + " " + HRvideofilename + " height=" + str(height) + " is not a multiple of the downscale factor: " + str(set.downscale_scalefactor))

				# determine downscale size
				downscalewidth  = int(math.ceil(float(width)/float(set.downscale_scalefactor)))
				downscaleheight = int(math.ceil(float(height)/float(set.downscale_scalefactor)))

				# determine actual algorithm used
				algorithm = set.downscale_algorithm
				if set.downscale_algorithm == 6:
					if set.downscale_random_per_scene and index > 0:
						algorithm = first_frame_algorithm
					else:
						algorithm = random.randrange(6)

				algorithm_names = ["bilinear", "nearest", "area", "bicubic", "Lanczos", "Blur"]
				downscale_algorithm.append((algorithm_names[algorithm]))
				if 0 <= algorithm <= 4:
					imResize = cv2.resize(im, dsize=(int(downscalewidth), int(downscaleheight)), interpolation=algorithm)
				elif algorithm == 5:
					# do a gaussian blur
					sigma = 1.5
					size = 9
					if set.downscale_scalefactor == 2:
						sigma = 0.75
						size = 5
					elif set.downscale_scalefactor == 4:
						sigma = 1.5
						size = 9

					blur = cv2.GaussianBlur(im, (size, size), sigma)
					imResize = cv2.resize(blur, dsize=(int(downscalewidth), int(downscaleheight)), interpolation=algorithm)

				quality = set.downscale_JPEG_quality
				if set.downscale_format.upper() == "PNG":
					cv2.imwrite(LRvideofilename, imResize)
				else:
					if quality == -1:
						if set.downscale_random_per_scene and index > 0:
							quality = first_frame_jpeg_quality
						else:
							quality = random.randrange(100)
					cv2.imwrite(LRvideofilename, imResize), [int(cv2.IMWRITE_JPEG_QUALITY), quality]

				# remember first frame random settings
				if index == 0:
					first_frame_algorithm = algorithm
					first_frame_jpeg_quality = quality

		if set.downscale_create_mkv:
			FNULL = open(os.devnull, 'w')
			inputfilter = os.path.join(LRfoldername, "%04d.png")
			mkvpath = os.path.join(LRfoldername, scene['folder_name'] + ".mkv")
			cmd0 = "ffmpeg -framerate 30 -i " + inputfilter + " -c:v copy " + mkvpath
			subprocess.call(cmd0, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
			FNULL.close()

		# save downscale info to downscale.json
		downscale_path = os.path.join(LRfoldername, "downscale.json")
		with open(downscale_path, 'w') as f:
			json.dump(downscale_algorithm, f, indent=2, separators=(',', ':'))

		logging.info("Downscaling scene " + str(scene["scene_index"]) + " to " + LRfoldername + " complete")

	def ExecuteStage(self):
		logging.info("StageDownscale..executing stage")

		if settings.set.downscale_skip_stage:
			logging.info("StageDownscale..skipping stage")
			return

		# create a folder name for Low Resolution folders
		LRfoldername = settings.set.absolute_LR_folder
		if not path.exists(LRfoldername):
			os.makedirs(LRfoldername)

		# get a list of videos in the database
		videos = database.db.getVideos()

		# for each video:
		sceneindices = database.db.getSceneIndices()

		if settings.set.downscale_format.upper() != "PNG" and settings.set.downscale_format.upper() != "JPEG":
			logging.warning("Unsupported downscale format: " + settings.set.downscale_format.upper()+ " Needs to be PNG or JPEG")
			return

		if settings.set.multiprocess:
			processes = []

			logging.info("StageDownscale..starting multiprocess..# of scenes = " + str(len(sceneindices)))

			# In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
			# Python interpreter is started for each new multiprocessing.Process.This means that all global variables
			# are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
			# the spawned processes.
			# https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
			setcopy = copy.deepcopy(settings.set)

			for sceneid in sceneindices:

				p = mp.Process(target=self.downscale, args=(setcopy, database.db.getScene(sceneid)))
				processes.append(p)

			[p.start() for p in processes]
			[p.join() for p in processes]

			# copy back
			settings.set = copy.deepcopy(setcopy)

		else:
			for sceneid in sceneindices:

				self.downscale(settings.set, database.db.getScene(sceneid))

		# set the lr_path
		for sceneid in sceneindices:

			scene = database.db.getScene(sceneid)
			scene['lr_path'] = os.path.join(settings.set.absolute_LR_folder, scene['folder_name'])

		database.db.save()
		logging.info("StageDownscale..complete for all scenes")