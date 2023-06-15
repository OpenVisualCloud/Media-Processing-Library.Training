# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
import database
import os
import os.path
from stage import Stage, StageEnum
import glob
import copy
import multiprocessing as mp
import coloredlogs
import utils

STAGE_NAME = 'StageAugment'

class StageAugment(Stage):

	def __init__(self):
		self.enum = StageEnum.AUGMENT
		pass

	def augment(self, scene, set):
		import cv2
		import numpy as np
		from imgaug import augmenters as iaa
		import json
		import random

		# Configure Logging
		logger = logging.getLogger(__name__)
		coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

		files = utils.GetFiles(scene['lr_path'])
		if len(files) == 0:
			logging.warning("WARNING: LR folder does not contain any files: " + scene['lr_path'])
			return
		augmented = False
		effects = []
		for index, file in enumerate(files):

			if index % 20 == 0:
				logging.info("StageAugment..augmenting scene " + str(scene['scene_index']) + ' ' + str(index) + " of " + str(len(files)) + " completed")

			if not os.path.exists((file)):
				logging.warning("WARNING.  file does not exist: "+ file)
				continue

			im: np.ndarray = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
			if im is None:
				logging.warning("WARNING.  unable to read file as image: "+ file)
				continue

			if set.augment_effect == "random":
				if len(set.augment_random_effects) > 0:
					augeffect = random.choice(set.augment_random_effects)
				else:
					logging.warning("WARNING.  no random augmentation effects defined")
					logging.warning("WARNING.  check spelling mistakes in augment_random_effects config option and remove any spaces in it.")
					return
			else:
				augeffect = set.augment_effect
			if 'jpegcompression' in augeffect and im.dtype.type == np.uint16:
				logging.warning("WARNING. jpegcompression only supported for 8bit images")
				continue

			if augeffect == "salt_and_pepper":
				augmentation = iaa.SaltAndPepper(0.001)
			elif augeffect == "gaussian_low":
				augmentation = iaa.AdditiveGaussianNoise(0, 2)
			elif augeffect == "gaussian_medium":
				augmentation = iaa.AdditiveGaussianNoise(0, 5)
			elif augeffect == "gaussian_high":
				augmentation = iaa.AdditiveGaussianNoise(0, 10)
			elif augeffect == "block_2":
				augmentation = iaa.AveragePooling(kernel_size=2, keep_size=True)
			elif augeffect == "block_4":
				augmentation = iaa.AveragePooling(kernel_size=4, keep_size=True)
			elif augeffect == "block_8":
				augmentation = iaa.AveragePooling(kernel_size=8, keep_size=True)
			elif augeffect == "block_16":
				augmentation = iaa.AveragePooling(kernel_size=16, keep_size=True)
			elif augeffect == "block_32":
				augmentation = iaa.AveragePooling(kernel_size=32, keep_size=True)
			elif augeffect == "block_64":
				augmentation = iaa.AveragePooling(kernel_size=64, keep_size=True)
			elif augeffect == "burst_low":
				augmentation = iaa.Sequential([
					iaa.Cutout(nb_iterations=(1, 3), size=(0.025, 0.100), squared=False, fill_mode="constant", cval=0),
					iaa.Cutout(nb_iterations=(1, 3), size=(0.025, 0.100), squared=False, fill_mode="constant",
							   cval=255)])
			elif augeffect == "burst_medium":
				augmentation = iaa.Sequential([
					iaa.Cutout(nb_iterations=(4, 10), size=(0.025, 0.100), squared=False, fill_mode="constant", cval=0),
					iaa.Cutout(nb_iterations=(4, 10), size=(0.025, 0.100), squared=False, fill_mode="constant",
							   cval=255)])
			elif augeffect == "burst_high":
				augmentation = iaa.Sequential([
					iaa.Cutout(nb_iterations=(12, 30), size=(0.025, 0.100), squared=False, fill_mode="constant",
							   cval=0),
					iaa.Cutout(nb_iterations=(12, 30), size=(0.025, 0.100), squared=False, fill_mode="constant",
							   cval=255)])
			elif augeffect == "poisson_low":
				augmentation = iaa.AdditivePoissonNoise(lam=(0, 10), per_channel=True)
			elif augeffect == "poisson_medium":
				augmentation = iaa.AdditivePoissonNoise(lam=(0, 20), per_channel=True)
			elif augeffect == "poisson_high":
				augmentation = iaa.AdditivePoissonNoise(lam=(0, 30), per_channel=True)
			elif augeffect == "laplace_low":
				augmentation = iaa.AdditiveLaplaceNoise(scale=(0, 0.02*255))
			elif augeffect == "laplace_medium":
				augmentation = iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255))
			elif augeffect == "laplace_high":
				augmentation = iaa.AdditiveLaplaceNoise(scale=(0, 0.10*255))
			elif augeffect == "jpegcompression_low":
				augmentation = iaa.JpegCompression(compression=(5, 25))
			elif augeffect == "jpegcompression_medium":
				augmentation = iaa.JpegCompression(compression=(26, 69))
			elif augeffect == "jpegcompression_high":
				augmentation = iaa.JpegCompression(compression=(70, 99))
			elif augeffect == "motionblur_low":
				augmentation = iaa.MotionBlur(k=3, angle=(0, 360))
			elif augeffect == "motionblur_medium":
				augmentation = iaa.MotionBlur(k=7, angle=(0, 360))
			elif augeffect == "motionblur_high":
				augmentation = iaa.MotionBlur(k=15, angle=(0, 360))
			elif augeffect == "sharpen_low":
				augmentation = iaa.Sharpen(alpha=(0.2), lightness=(1.0))
			elif augeffect == "sharpen_medium":
				augmentation = iaa.Sharpen(alpha=(0.4), lightness=(1.0))
			elif augeffect == "sharpen_high":
				augmentation = iaa.Sharpen(alpha=(0.6), lightness=(1.0))
			else:
				logging.warning("WARNING: unsupported augmentaton effect: " + set.augment_effect)
				break
			effects.append(augeffect)
			im = augmentation(image=im)
			cv2.imwrite(file, im)
			augmented = True

		# save downscale info to downscale.json
		if augmented:
			augment_path = os.path.join(scene['lr_path'], "augment.json")
			with open(augment_path, 'w') as f:
				json.dump(effects, f, indent=2, separators=(',', ':'))


		logging.info("StageAugment..augmenting scene " + str(scene['scene_index']) + " finished.")

	def ExecuteStage(self):
		logging.info("StageAugment...executing stage")

		if settings.set.augment_skip_stage:
			logging.info("StageAugment..skipping stage")
			return

		if settings.set.augment_effect == "":
			logging.warning("WARNING: No augmentation enabled")
			return

		sceneindices = database.db.getSceneIndices()
		if settings.set.multiprocess:
			processes = []

			logging.info("StageAugment..starting multiprocess..# of scenes = " + str(len(sceneindices)))

			# In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
			# Python interpreter is started for each new multiprocessing.Process.This means that all global variables
			# are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
			# the spawned processes.
			# https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
			dbcopy = copy.deepcopy(database.db)
			setcopy = copy.deepcopy(settings.set)

			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageAugment..starting process for scene=" + str(sceneid))

				p = mp.Process(target=self.augment, args=(database.db.getScene(sceneid), setcopy))
				processes.append(p)

			[p.start() for p in processes]
			[p.join() for p in processes]
			logging.info("StageAugment..multiprocess complete")

			# copy back
			settings.set = copy.deepcopy(setcopy)
			database.db = copy.deepcopy(dbcopy)

		else:
			logging.info("StageAugment..starting to process scenes..# of scenes=" + str(len(sceneindices)))

			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageSplit..starting process for scene=" + str(sceneid))

				self.augment(database.db.getScene(sceneid), settings.set)

			logging.info("StageAugment Complete")
