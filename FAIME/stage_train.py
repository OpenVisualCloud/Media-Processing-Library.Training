# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
from stage import Stage, StageEnum
import settings
import database
import os
import os.path
from os import path
import subprocess
import platform
import zipfile
import warnings
import glob
import utils
warnings.filterwarnings("ignore")

import shutil


def preexec():  # Don't forward signals.
	os.setpgrp()

def folder_check(path):
	try_num = 1
	oripath = path[:-1] if path.endswith('/') else path
	while os.path.exists(path):
		print("Delete existing folder " + path + "?(Y/N)")
		decision = input()
		if decision == "Y":
			shutil.rmtree(path, ignore_errors=True)
			break
		else:
			path = oripath + "_%d/" % try_num
			try_num += 1
			print(path)

	return path

def mycall(cmd, block=False):
	if platform.system() == "Windows" or (not block):
		return subprocess.Popen(cmd)
	else:
		return subprocess.Popen(cmd, preexec_fn=preexec)

def sharpen(hr_images, sharpen_path):
	# Import the Path class from the pathlib module
	from pathlib import Path

	# Use Raisr sharpen for Sharpening 10bit images
	if settings.set.train_raisr_bit_depth == '10':
		# Import the Sharpen class from the Raisr module
		from Raisr import Sharpen

		# Set up the sharpen parameters
		sharpen_params = [
			'-bits', settings.set.train_raisr_bit_depth,
			'-o', sharpen_path,
			'-s', str(settings.set.train_raisr_sharpen)
		]

		# Use the Sharpen class to sharpen the HR images
		Sharpen.main({'hr_images':hr_images},sharpen_params)

	# Use FFmpeg for Sharpening 8bit images
	elif settings.set.train_raisr_bit_depth == '8':
		# Import the ffmpeg module
		import ffmpeg

		# Try to sharpen the HR images using ffmpeg
		try:
			# Loop through each HR image
			for file in hr_images:
				# Get the base path of the new file in the sharpen_path
				newfile_basepath = os.path.join(
						os.path.basename(os.path.dirname(file)),
						os.path.basename(file))

				# Create the folder for the new file in the sharpen_path
				Path(os.path.join(sharpen_path,os.path.dirname(newfile_basepath))
						).mkdir(parents=True,exist_ok=True)

				# Use ffmpeg to sharpen the file and output it to the new file path
				(ffmpeg
					.input(file)
					.filter('unsharp',5,5,settings.set.train_raisr_sharpen,5,5,0.0)
					.output(os.path.join(sharpen_path,newfile_basepath))
					.run(quiet=True,overwrite_output=True)
				)
		except ffmpeg._run.Error as e:
			# Print any error messages from ffmpeg
			logging.warning(f'ffmpeg error: {e.stderr}')



class StageTrain(Stage):

	def __init__(self):
		self.enum = StageEnum.TRAIN
		pass

	def LogVariable(scope,key): # key=tf.GraphKeys.MODEL_VARIABLES):
		import tensorflow as tf
		import numpy as np
		logging.info("Scope %s:" % scope)
		variables_names = [[v.name, v.get_shape().as_list()] for v in tf.get_collection(key, scope=scope)]
		total_sz = 0
		for k in variables_names:
			logging.info("Variable: " + k[0])
			logging.info("Shape: " + str(k[1]))
			total_sz += np.prod(k[1])
		logging.info("total size: %d" % total_sz)

	def TrainRaisr(self):
		import Raisr.Train as Raisr
		from pathlib import Path

		passed_args = {}
		passed_args['scale_factor'] = settings.set.train_raisr_scale
		passed_args['patch_size'] = settings.set.train_raisr_patch_size
		passed_args['gradient_size'] = settings.set.train_raisr_gradient_size
		passed_args["angle_quantization"] = settings.set.train_raisr_angle_quantization
		passed_args["strength_quantization"] = settings.set.train_raisr_strength_quantization
		passed_args["coherence_quantization"] = settings.set.train_raisr_coherence_quantization
		config_contents = [passed_args["angle_quantization"],passed_args["strength_quantization"],passed_args["coherence_quantization"],passed_args['patch_size']]
		filters_path = os.path.join(settings.set.absolute_train_folder,"Raisr",settings.set.train_raisr_filterpath)
		# quant_path = os.path.join(settings.set.absolute_train_folder, "Raisr","quantization")


		Path(filters_path).mkdir(parents=True, exist_ok=True)
		#Path(quant_path).mkdir(parents=True,exist_ok=True)

		passed_args["hr_images"] = []
		passed_args["chr_images"] = []
		eight_bit = []
		eight_bit_lr = []
		eight_bit_chr = []
		ten_bit = []
		ten_bit_lr = []
		ten_bit_chr = []


		# Train RAISR filter with different HR files than those that exist in the database
		if settings.set.train_raisr_input_folder:
				if not os.path.exists(settings.set.train_raisr_input_folder):
					logging.warning("StageTrain..WARNING - raisr input folder does not exist: " + settings.set.train_raisr_input_folder)
					return
				inputfiles = utils.GetFiles(settings.set.train_raisr_input_folder)
				if settings.set.train_raisr_bit_depth == '8':
					eight_bit = inputfiles
				elif settings.set.train_raisr_bit_depth == '10':
					ten_bit = inputfiles
				else:
					logging.warning('StageTrain..WARNING - Bit depth value of {} is not valid'.format(settings.set.train_raisr_bit_depth))
					return

				if len(eight_bit) == 0 and len(ten_bit) == 0:
					logging.warning("StageTrain..WARNING - raisr input folder does not contain any png files: " + settings.set.train_raisr_input_folder)
					return
		# Train RAISR filter with database HR files
		else:
			sceneindices = database.db.getSceneIndices()
			if len(sceneindices) == 0:
				logging.warning("StageTrain..WARNING - no scenes to process")
				return

			# Use HR images
			for sceneindex in sceneindices:

				logging.info("StageTrain.starting processing scene " + str(sceneindex) + " of " + str(len(sceneindices)))
				scene = database.db.getScene(sceneindex)

				# if individual scenes are specified
				if len(settings.set.individual_scenes) > 0 and sceneindex not in settings.set.individual_scenes:
					logging.info("StageTrain...skipping scene " + str(sceneindex) + " (not in individual_scenes list)")
					continue

				# Get the folder containing the high-resolution images
				# If the scene is cropped, use the cropped folder
				# Otherwise, use the high-resolution folder
				hr_folder = scene['cropped_path'] if scene['cropped'] else scene['hr_path']
				# Get the files in the high-resolution folder
				hr_files = utils.GetFiles(hr_folder)

				# Get the folder containing the low-resolution images, if it exists
				lr_folder = scene['lr_path']
				lr_files = utils.GetFiles(lr_folder) if lr_folder else []

				# Get the folder containing the cheap upscale high-resolution images, if it exists
				chr_folder = scene['chr_path']
				chr_files = utils.GetFiles(chr_folder) if chr_folder else []

				# Add the high-resolution files and low-resolution files to the appropriate sets
				if scene['bit_depth'] == '8':
					eight_bit.extend(hr_files)
					eight_bit_lr.extend(lr_files)
					eight_bit_chr.extend(chr_files)
				elif scene['bit_depth'] == '10':
					ten_bit.extend(hr_files)
					ten_bit_lr.extend(lr_files)
					ten_bit_chr.extend(chr_files)
				elif settings.set.train_raisr_bit_depth == '8':
					eight_bit.extend(hr_files)
					eight_bit_lr.extend(lr_files)
					eight_bit_chr.extend(chr_files)
				elif settings.set.train_raisr_bit_depth == '10':
					ten_bit.extend(hr_files)
					ten_bit_lr.extend(lr_files)
					ten_bit_chr.extend(chr_files)

		# First pass always trains with LR into HR
		if settings.set.train_raisr_first_pass:
			# Define the parameters for the Raisr.main() function
			parameters = ['-ff', filters_path]

			logging.info('StageTrain..Training first pass of {} 8 bit files and {} 10 bit files'.format(len(eight_bit),len(ten_bit)))

			# Loop over the 8-bit and 10-bit input files
			for bit_depth, hr_files, lr_files in [(8, eight_bit, eight_bit_lr), (10, ten_bit, ten_bit_lr)]:
				# Skip this bit depth if there are no input files
				if not hr_files:
					continue
				logging.info('StageTrain..Training first pass of {} bit files '.format(bit_depth))
				# Set the bit depth parameter
				parameters += ['-bits', str(bit_depth)]

				# Set the low-resolution images parameter if there are low-resolution files
				if lr_files:
					parameters += ['-lf', 'True']
					passed_args['lr_images'] = lr_files
				else:
					passed_args['lr_images'] = []

				# Set the high-resolution images parameter and call Raisr.main()
				passed_args['hr_images'] = hr_files
				Raisr.main(passed_args, parameters)

		# Code modelled to replicate below Python call to Train.py in RAISR
		# Train.py -ohf HR_Sharp -chf HR -bits 10 -ff filter_sharp
		# Second pass trains HR images into Sharp HR images
		if settings.set.train_raisr_second_pass:
			# Define paths for the sharpen and filters_sharp directories
			sharpen_path = os.path.join(settings.set.absolute_train_folder, 'Raisr', 'sharpen')
			if os.path.exists(sharpen_path):
				shutil.rmtree(sharpen_path)
			Path(sharpen_path).mkdir(parents=True, exist_ok=True)

			sharp_filters_path = os.path.join(settings.set.absolute_train_folder, "Raisr", "filters_sharp")
			Path(sharp_filters_path).mkdir(parents=True, exist_ok=True)
			chr_folder = chr_folder if chr_folder!="" else hr_folder
			# Define the parameters for the Raisr.main() function
			parameters = ['-ff', sharp_filters_path, '-chf', chr_folder]
			logging.info('StageTrain..Training second pass of {} 8 bit files and {} 10 bit files'.format(len(eight_bit),len(ten_bit)))
			# Loop over the 8-bit and 10-bit input files
			for bit_depth, hr_files,chr_files in [(8, eight_bit,eight_bit_chr), (10, ten_bit,ten_bit_chr)]:
				# Skip this bit depth if there are no input files
				if not hr_files:
					continue
				logging.info('StageTrain..Training second pass of {} bit files '.format(bit_depth))
				# Set the bit depth parameter
				parameters += ['-bits', str(bit_depth)]

				# Sharpen the high-resolution images
				sharp_path = os.path.join(sharpen_path, '{}_bit'.format(bit_depth))
				Path(sharp_path).mkdir(parents=True, exist_ok=True)
				sharpen(hr_files, sharp_path)

				# Set the sharpened high-resolution images as the target images
				# and the original high-resolution images as the input images
				if chr_files:
					passed_args['chr_images'], passed_args['hr_images'] = chr_files, utils.GetFiles(sharp_path,recursive=True)
				else:
					passed_args['chr_images'], passed_args['hr_images'] = hr_files, utils.GetFiles(sharp_path,recursive=True)

				passed_args['lr_images'] = []

				# Call Raisr.main() to train the filters
				Raisr.main(passed_args, parameters)

				# Move the trained filters to the filters directory
				bin_files = glob.glob(os.path.join(sharp_filters_path, '*bin*'))
				for file in bin_files:
					base_file = os.path.basename(file)
					shutil.move(file, os.path.join(filters_path, base_file + '_2'))
			# Delete the filters_sharp directory
			shutil.rmtree(sharp_filters_path)
		config_path = os.path.join(filters_path, 'config')
		if not os.path.exists(config_path):
			open(config_path,'w').write(' '.join(map(str,config_contents)))
		else:
			os.remove(config_path)
			open(config_path,'w').write(' '.join(str(n) for n in config_contents))
		pass

	def ExecuteStage(self):

		logging.info("StageTrain..executing stage")

		if settings.set.train_skip_stage:
			logging.info("StageTrain..skipping stage")
			return

		if settings.set.train_algorithm.lower() == "raisr":
			self.TrainRaisr()
		else:
			logging.warning("WARNING: Unsupported train algorithm: " + settings.set.train_algorithm)
