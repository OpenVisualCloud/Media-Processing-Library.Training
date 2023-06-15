# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import coloredlogs
import os
import os.path
from os import path
from stage import Stage, StageEnum
import settings
import database
import video
import errno
import multiprocessing as mp
import copy
from PIL import Image, ImageDraw, ImageFont
import math
import json
from FAIME import utils
STAGE_NAME = 'StageSplit'

class StageSplit(Stage):

	def __init__(self):
		self.enum = StageEnum.SPLIT
		pass

	# adapted from https://code.activestate.com/recipes/412982-use-pil-to-make-a-contact-sheet-montage-of-images/
	def make_contact_sheet(self, title, fnames, values_list, values_range_list, values_name_list, first_scene_frame, ncols, nrows, frame_width, frame_height,
						   margin_left, margin_top, margin_right, margin_bottom,
						   padding):
		"""
        Make a contact sheet from a group of filenames:

        fnames       A list of names of the image files

        ncols        Number of columns in the contact sheet
        nrows        Number of rows in the contact sheet
        photow       The width of the photo thumbs in pixels
        photoh       The height of the photo thumbs in pixels

        marl         The left margin in pixels
        mart         The top margin in pixels
        marr         The right margin in pixels
        marl         The left margin in pixels

        padding      The padding between images in pixels

        returns a PIL image object.
        """
		LARGE_FONT_SIZE = 20
		SMALL_FONT_SIZE = 8

		SECOND_RECT_INSET = 4

		NUM_DESCRIPTION_LINES = 3

		# Read in all images/frames and resize appropriately
		imgs = [Image.open(fn).resize((frame_width, frame_height)) for fn in fnames]

		# title appears at the top of image
		TITLEHEIGHT = 30

		# descriptions appear below the photos
		DESCRIPTION_HEIGHT = 12

		# http://web-tech.ga-usa.com/2012/05/creating-a-custom-hot-to-cold-temperature-color-gradient-for-use-with-rrdtool/index.html
		# hot to cold olor scale
		COLOR_SCALE = [(0,0,255), (0,255,244), (101,255,0), (255,240,0), (255,90,0), (255, 0, 64),(255, 14, 240)]
		WHITE_COLOR = (255, 255, 255)

		# compute various dimensions (margins, etc)
		marw = margin_left + margin_right
		marh = margin_top + margin_bottom
		padw = (ncols - 1) * padding

		# determine overall width/height of the image
		imagewidth = ncols * frame_width + marw + padw
		imageheight = TITLEHEIGHT + marh + nrows * (frame_height+padding + NUM_DESCRIPTION_LINES*DESCRIPTION_HEIGHT)
		isize = (imagewidth, imageheight)

		# Create the new image. The background doesn't have to be white
		inew = Image.new('RGB', isize, WHITE_COLOR)

		# create the fonts.  Will use FreeSans.ttf in fonts-freefont-ttf
		font_path = os.path.join("FAIME","resources","FreeSans.ttf")
		font = ImageFont.truetype(font_path, LARGE_FONT_SIZE)
		fontsmall = ImageFont.truetype(font_path, SMALL_FONT_SIZE)

		# draw lines and text for each frame
		numframes = len(fnames)

		# create a draw object to draw text and lines on
		draw = ImageDraw.Draw(inew)

		# draw title
		sizew, sizeh = draw.textsize(title, font)
		draw.text((int(imagewidth/2 - sizew/2),int(TITLEHEIGHT/2 - sizeh/2)), title, (0,0,0), font=font, align="ms")

		# draw lines and text for each frame
		frameindex = 0

		value_increment = values_range_list[0]/len(COLOR_SCALE)
		for irow in range(nrows):
			for icol in range(ncols):
				if frameindex < numframes:

					left = margin_left + icol * (frame_width + padding)
					right = left + frame_width
					upper = TITLEHEIGHT + margin_top + irow * (frame_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
					lower = upper + frame_height
					bbox = (left, upper, right, lower)
					try:
						img = imgs.pop(0)
					except:
						break
					inew.paste(img, bbox)

					value = values_list[0][frameindex]

					# compute dimensions for the frame
					left = margin_left + icol * (frame_width + padding)
					right = left + frame_width
					upper = TITLEHEIGHT + margin_top + irow * (frame_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
					lower = upper + frame_height
					bbox = (left, upper,right,lower)

					# Determine the color of the surrounding outline
					# Note: value could be None at frame 0 where is no previous frame to calculate a delta frame.
					# In that case set to the maximum color in the scale
					if value == None:
						colorindex = len(COLOR_SCALE) - 1
					else:
						colorindex = (int)(value / value_increment )
						if colorindex >= len(COLOR_SCALE):
							colorindex = len(COLOR_SCALE) - 1
					color = COLOR_SCALE[colorindex]

					# draw a colored rectangle around the frame
					draw.rectangle(bbox, outline=color, fill=None, width=2)

					# draw a second colored rectangle around the frame
					if len(values_range_list) > 1:
						value2 = values_list[1][frameindex]
						value2_increment = values_range_list[1] / len(COLOR_SCALE)
						# Determine the color of the surrounding outline
						# Note: value could be None at frame 0 where is no previous frame to calculate a delta frame.
						# In that case set to the maximum color in the scale
						if value2 == None:
							colorindex = len(COLOR_SCALE) - 1
						else:
							colorindex = (int)(value2 / value2_increment )
							if colorindex >= len(COLOR_SCALE):
								colorindex = len(COLOR_SCALE) - 1
						color2 = COLOR_SCALE[colorindex]
						bbox2 = (left+SECOND_RECT_INSET, upper+SECOND_RECT_INSET, right-SECOND_RECT_INSET, lower-SECOND_RECT_INSET)
						# draw a second colored rectangle around the frame
						draw.rectangle(bbox2, outline=color2, fill=None, width=2)

					# Description line 1- draw the frame number
					center = int((left + right)/2)
					middle = int(lower + DESCRIPTION_HEIGHT/2)
					desc = str(frameindex) + "(" + str(first_scene_frame + frameindex) + ") "
					sizew, sizeh = draw.textsize(desc, fontsmall)
					draw.text((int(center-sizew/2), int(middle-sizeh/2)), desc, (0,0,0), font = fontsmall, align="ms")

				    # Description line 2 - draw the first value
					middle = int(lower + DESCRIPTION_HEIGHT*3/2)
					if value == None:
						desc =  values_name_list[0] + ": infinite"
					else:
						desc = values_name_list[0] + ": " + "{:7.4f}".format(value)
					sizew, sizeh = draw.textsize(desc, fontsmall)
					draw.text((int(center-sizew/2), int(middle-sizeh/2)), desc, (0,0,0), font = fontsmall, align="ms")

					# Description line 3 - draw the second value
					if len(values_list) > 1:
						middle = int(lower + DESCRIPTION_HEIGHT * 5 / 2)
						value2 = values_list[1][frameindex]
						if value2 == None:
							desc = values_name_list[1] + ": infinite"
						else:
							desc = values_name_list[1] + ": " + "{:7.4f}".format(value2)
						sizew, sizeh = draw.textsize(desc, fontsmall)
						draw.text((int(center-sizew/2), int(middle-sizeh/2)), desc, (0,0,0), font = fontsmall, align="ms")

				frameindex = frameindex + 1
		# return the image
		return inew

	def splitscene(self, scene, HRfolder, contactsheetfolder, set):
		import cv2

		# Configure Logging
		logger = logging.getLogger(__name__)
		coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

		# define some parameters for the contact sheet
		CONTACT_SHEET_FRAME_WIDTH = 100
		CONTACT_SHEET_FRAME_MARGIN = 2
		CONTACT_SHEET_PADDING = 10
		CONTACT_SHEET_FRAMES_PER_ROW = 10

		create_contactsheet = True

		logging.info("Processing Scene: " + str(scene['scene_index']))

		# start processing the scenes in the video
		first_scene_frame = scene['start_frame']
		last_scene_frame = scene['end_frame']
		frame_width = scene['frame_width']
		frame_height = scene['frame_height']
		frame_aspect = float(frame_height)/float(frame_width)
		contact_sheet_width = int(CONTACT_SHEET_FRAME_WIDTH)
		contact_sheet_height = int(CONTACT_SHEET_FRAME_WIDTH*frame_aspect)
		folder_name = scene['folder_name']

		# Create a folder for the scene
		splitfoldername = os.path.join(HRfolder, folder_name)
		if path.exists(splitfoldername) and not set.split_overwrite:
			logging.warning("WARNING: Scene folder exists: " + splitfoldername )
			return
		if not path.exists(splitfoldername):
			os.makedirs(splitfoldername)

		# create lists of filenames and metric values to use when creating the contact sheet
		framenames = []
		values = []
		values2 = []

		# INTER_NEAREST – a nearest-neighbor interpolation
		# INTER_LINEAR – a bilinear interpolation (used by default)
		# INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
		# INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
		interpolationalg = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

		# load up the video and get the number of frames
		video_path = os.path.join(set.absolute_videos_folder, scene['video_name'])

		if not os.path.exists(video_path):
			logging.warning("WARNING: video for scene:" + scene['video_name'] + " doesn't exists: " + video_path)
			return

		video_capture = video.Video(video_path, STAGE_NAME)

		# read the scenedetection metrics
		scenedetection_metrics = []
		scenedetection_metrics_path = os.path.join(set.absolute_videos_folder, os.path.splitext(scene['video_name'])[0]+ "_scenedetection.json")
		if not os.path.exists(scenedetection_metrics_path):
			logging.warning("WARNING: scene detection metrics file not found: "+ scenedetection_metrics_path)
		else:
			with open(scenedetection_metrics_path, 'r') as f:
				jsondata = json.load(f)
				scenedetection_metrics = jsondata['scenedetection_metrics']

		values, values2 = video_capture.SplitScene(scene, splitfoldername, scenedetection_metrics, set)
		# create title for contact sheet
		if create_contactsheet:
			title = "Scene " + str( scene['scene_index']) + " from: " + scene['video_name'] + "(frames: " + str(first_scene_frame) + "-" + str(
			last_scene_frame - 1) + ")"

			values_list = []
			values_range_list = []
			values_name_list = []
			if scene['detection_alg'] == 'histogram':
				values_list.append(values)
				values_list.append(values2)

				values_range_list.append(set.scenedetect_histogram_threshold)
				values_range_list.append(set.scenedetect_histogram_nei_threshold)
				values_name_list.append('hist diff')
				values_name_list.append('hist nei diff')

			else:
				values_list.append(values)
				values_range_list.append(255)
				values_name_list.append('dhsv')
			framenames = utils.GetFiles(os.path.join(settings.set.absolute_HR_folder, scene['folder_name']))
			img = self.make_contact_sheet(title, framenames, values_list, values_range_list, values_name_list,
										  first_scene_frame,
										  CONTACT_SHEET_FRAMES_PER_ROW,
										  math.ceil(set.scenedetect_max_accept_length / CONTACT_SHEET_FRAMES_PER_ROW),
										  contact_sheet_width, contact_sheet_height,
										  CONTACT_SHEET_FRAME_MARGIN, CONTACT_SHEET_FRAME_MARGIN,
										  CONTACT_SHEET_FRAME_MARGIN, CONTACT_SHEET_FRAME_MARGIN,
										  CONTACT_SHEET_PADDING)

			filename = os.path.join(contactsheetfolder, folder_name + ".png")
			img.save(filename)

		pass

	def ExecuteStage(self):
		logging.info("StageSplit..executing stage")

		if settings.set.split_skip_stage:
			logging.info("StageSplit..skipping stage")
			return

		# create the contact sheet folder
		contactsheetfolder = settings.set.absolute_contact_sheets_folder
		if not path.exists(contactsheetfolder):
			os.makedirs(contactsheetfolder)

		if len(database.db.scenes)== 0:
			logging.warn("WARNING no scenes found to split")
			return

		sceneindices = database.db.getSceneIndices()
		if settings.set.multiprocess:
			processes = []

			logging.info("StageSplit..starting multiprocess..# of scenes = " + str(len(sceneindices)))

			# In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
			# Python interpreter is started for each new multiprocessing.Process.This means that all global variables
			# are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
			# the spawned processes.
			# https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
			dbcopy = copy.deepcopy(database.db)
			setcopy = copy.deepcopy(settings.set)

			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageSplit..starting process for scene=" + str(sceneid))

				p = mp.Process(target=self.splitscene, args=(database.db.getScene(sceneid), settings.set.absolute_HR_folder, contactsheetfolder, setcopy))
				processes.append(p)

			[p.start() for p in processes]
			[p.join() for p in processes]
			logging.info("StageSplit..multiprocess complete")

			# copy back
			settings.set = copy.deepcopy(setcopy)
			database.db = copy.deepcopy(dbcopy)

		else:
			logging.info("StageSplit..starting to process videos..# of scenes="  + str(len(sceneindices)))

			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageSplit..starting process for scene=" + str(sceneid))

				self.splitscene(database.db.getScene(sceneid), settings.set.absolute_HR_folder, contactsheetfolder, settings.set)

		for sceneid in sceneindices:
			scene = database.db.getScene(sceneid)
			# remove the frame_dhsv array from the scenes in the database to clean it up
			scene['frame_dhsv'] = []

			# set the high resolution path
			scene['hr_path'] = os.path.join(settings.set.absolute_HR_folder, scene['folder_name'])

		database.db.save()

		logging.info("StageSplit..splitting complete for all scenes")
