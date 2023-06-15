# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import settings
import logging
import coloredlogs
import os
import glob
from stage import Stage, StageEnum
import multiprocessing as mp
import database
import copy
import json


DEFAULT_DOWNSCALE_FACTORS = {
    3200: 12,   # ~4k
    2100:  8,   # ~2k
    1700:  6,   # ~1080p
    1200:  5,
    900:   4,   # ~720p
    600:   3,
    400:   2    # ~480p
}

STAGE_NAME = 'StageSceneDetect'

class StageSceneDetect(Stage):

	def __init__(self):
		self.enum = StageEnum.SCENEDETECT
		pass

	def detect(self, filename, set):
		import cv2
		import datetime
		import numpy as np
		import video as vid
		# Standard PySceneDetect imports:
		from scenedetect import VideoManager
		from scenedetect import SceneManager
		from scenedetect import StatsManager

		# For content-aware scene detection:
		from scenedetect.detectors import ContentDetector
		from scenedetect.detectors import ThresholdDetector

		# Configure Logging
		logger = logging.getLogger(__name__)
		coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

		head, tail = os.path.split(filename)

		# Does this video contain a high percentage of duplicated frames - if so, it may have been created by duplicating frames to achieve a higher frame rate
		duplicated_frames = False

		if set.scenedetect_algorithm == 'histogram':
			video = vid.Video(filename, STAGE_NAME)
			if not video.isOpenable():
				logging.warning('Unable to open video file: ', filename)
				return
			before_open_ts = datetime.datetime.now()
			video.open()
			after_open_ts = datetime.datetime.now()
			if not video.isOpen():
				logging.warning('Failed to open video file: ', filename)
				return
			logging.info('StageSceneDetect opening video took {} seconds'.format((after_open_ts - before_open_ts).total_seconds()))
			numframes = int(video.frames)
			fps = int(video.fps)
			# adjust numframes to process if max video length setting is set
			if set.scenedetect_max_video_length != -1:
				numframes = min(numframes, set.scenedetect_max_video_length)

			# get the frame size
			framewidth = int(video.width)
			frameheight = int(video.height)

			# downscale the frame to speed up the scene detection logic
			if set.scenedetect_downscale == 0:
				scalefactor = 1
				for width in sorted(DEFAULT_DOWNSCALE_FACTORS, reverse=True):
					if framewidth >= width:
						scalefactor = DEFAULT_DOWNSCALE_FACTORS[width]
						break
				downscalewidth  =  framewidth / scalefactor
				downscaleheight =  frameheight / scalefactor
			else:
				downscalewidth  =  framewidth / set.scenedetect_downscale
				downscaleheight =  frameheight / set.scenedetect_downscale

			# histogram filtering kernel - center bin weighed the largest, weights sum to 1
			filtering_kernel = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]

			# define various histogram values
			histogram_block_size = set.scenedetect_histogram_block_size
			histogram_pixel_count = histogram_block_size * histogram_block_size
			histogram_num_bins = set.scenedetect_histogram_num_bins
			num_blocks_x = int(downscalewidth  // histogram_block_size)
			num_blocks_y = int(downscaleheight // histogram_block_size)
			num_blocks_total = int(num_blocks_x * num_blocks_y)
			histogram_distance_threshold = set.scenedetect_histogram_threshold
			histogram_nei_distance_threshold = set.scenedetect_histogram_nei_threshold

			# per frame histogram differences
			dif = []

			# determine the number of duplicated frames
			num_duplicated_frames = 0

			# determinte the max pixel value depending on bit depth
			max_pixel_value = 65535.0 if video.is10bit else 255.0

			# for each frame
			for j in range(numframes):

				# log progress
				if j%1000 == 0:
					logging.info("StageSceneDetect detecting " + filename + " frame "+str(j) + " of "+ str(numframes))

				# read the frame
				ret, frame = video.read()

				# see if frame is identical to previous frame
				if j > 0:
					comparison = frame == prev_frame
					equal_arrays = comparison.all()
					if equal_arrays:
						num_duplicated_frames = num_duplicated_frames + 1

				# convert to grayscale
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# downscale gray channel
				gray = cv2.resize(gray,
									  dsize=(int(downscalewidth),
											 int(downscaleheight)),
									  interpolation=cv2.INTER_LINEAR)

				# create a list of histograms for the values in the grayscale image; one histogram per block
				histogram = [[np.histogram(gray[y * histogram_block_size:(y + 1) * histogram_block_size,
								  x * histogram_block_size:(x + 1) * histogram_block_size],
								  bins=histogram_num_bins,
								  range=(0.0, max_pixel_value))[0] for x in range(num_blocks_x)] for y in range(num_blocks_y)]

				# create a list of differences with the current frame histogram and the previous frame
				if j == 0:
					# first frame
					hist_diff = [[0 for _ in range(num_blocks_x)]
								 for _ in range(num_blocks_y)]
				else:
					# non-first frame
					# N = length of kernel
					# convolve with kernel - Mode ‘same’ returns output of length max(M, N). Boundary effects are still visible.
					hist_diff = [[np.fabs(np.convolve(histogram[y][x] - prev_histogram[y][x], filtering_kernel, 'same'))
									for x in range(num_blocks_x)]
								 	for y in range(num_blocks_y)]

				# compute the average block histogram difference within the frame
				dif.append(np.sum(hist_diff) / (histogram_pixel_count * num_blocks_total))

				# remember the previous histogram
				prev_histogram = histogram

				prev_frame = frame

			video.close()
			# determine if there are a significant # of duplicated frames in the video
			if float(num_duplicated_frames)/float(numframes) > 0.25:
				duplicated_frames = True

			# calculate scene changes from the dif values
			scene_changes = [0]
			histogram_metrics = []
			neighborhoodsize = set.scenedetect_histogram_neighbor_size*2

			# note: minimum detection length is different from minimum acceptance length
			min_detection_length = 15

			# go through the frames
			for j in range(0, numframes):

				# create a list of  valid sequential frame indices (before and after the current frame)
				# this list is the neighborhood around the frame to use for calculating an average neighborhood distance
				indices = []
				for n in range(set.scenedetect_histogram_neighbor_size):
					indices.append(max(0, j - n))
					indices.append(min(numframes - 1, j + n))

				# calculate an average distance of the differences between the current frame and neighboring frames
				neighborhood_diff = 0.0
				for k in range(neighborhoodsize):
					neighborhood_diff += np.fabs(dif[j] - dif[indices[k]])
				neighborhood_diff /= neighborhoodsize

				# determine if this frame marks a scene change using the histogram difference and the neighborhood difference
				if neighborhood_diff > histogram_nei_distance_threshold and dif[j] > histogram_distance_threshold:
					scene_changes.append(j)
				histogram_metrics.append([dif[j], neighborhood_diff])

			# add the last frame  as the last scene change
			if len(scene_changes) == 0 or numframes - 1 - scene_changes[-1] > min_detection_length:
				scene_changes.append(numframes)
			else:
				scene_changes[-1] = numframes

			# construct scenelist
			scenelist = []
			lastsceneframe = -1
			firstsceneframe = scene_changes[0]
			scd_index = 1
			for j in range(numframes):
				if j == firstsceneframe:
					lastsceneframe = scene_changes[scd_index]-1

					# make sure the detected scene changes are not less than the minimum detection length
					if (lastsceneframe - firstsceneframe+ 1) >= min_detection_length:
						# append first scene frame, last scene frame, first scene timestamp, last scene timestamp
						scenelist.append([firstsceneframe, lastsceneframe, "", ""])

					scd_index += 1
					if scd_index >= len(scene_changes):
						break
				elif j == lastsceneframe:
					firstsceneframe = lastsceneframe+1
		else:
			# PySceneDetect

			# no easy way to detect duplicated frames, so just set to false
			duplicated_frames = True

			# Create the stats manager (recreate for every video)
			stats_manager = StatsManager()

			# Create the scene manager(recreate for every video)
			scene_manager = SceneManager(stats_manager)

			# Create the Content Detector specifying a threshold (avg_hsv) and a minimum detection length (recreate for every video)
			if set.scenedetect_algorithm == 'content':
				scene_manager.add_detector(ContentDetector(
					threshold=set.scenedetect_threshold,
					min_scene_len=set.scenedetect_min_detect_length))
			else:
				scene_manager.add_detector(ThresholdDetector(threshold=set.scenedetect_threshold))

			# Create the video manager for this video
			video_manager = VideoManager([filename])

			# Set the downscale factor (used to speed up detection)
			if set.scenedetect_downscale == 0:
				video_manager.set_downscale_factor(None)
			else:
				video_manager.set_downscale_factor(set.scenedetect_downscale)

			# Base timestamp at frame 0 (required to obtain the scene list).
			base_timecode = video_manager.get_base_timecode()

			#video_manager.start()
			video_manager._started = True
			video_manager._get_next_cap()
			scene_manager.detect_scenes(
				frame_source=video_manager,
				frame_skip=0
				# show_progress=False
			)
			scenelist = []
			scenes = scene_manager.get_scene_list(base_timecode)
			for ind, tup in enumerate(scenes):
				startframe = tup[0].get_frames()
				endframe = tup[1].get_frames()
				scenelist.append([startframe, endframe, tup[0].get_timecode(),tup[1].get_timecode()  ])

			framewidth = video_manager.get_framesize()[0]
			frameheight = video_manager.get_framesize()[1]
			fps = video_manager.get_framerate()
			numframes = video_manager._frame_length
			metricskeys = ['content_val', 'delta_hue', 'delta_sat', 'delta_lum']

			histogram_metrics = []
			for j in range(numframes):
				# get the metrics for this frame
				metrics = stats_manager.get_metrics(0, metricskeys)
				histogram_metrics.append([metrics[0], metrics[1], metrics[2], metrics[3]])

		# save the scene detection metrics to a file
		scenedetection_metrics_path = os.path.join(set.absolute_videos_folder, os.path.splitext(tail)[0]+ "_scenedetection.json")
		with open(scenedetection_metrics_path, 'w') as f:
			json.dump({"scenedetection_metrics": histogram_metrics}, f, indent=2, separators=(',', ':'))

		metricskeys = ['content_val', 'delta_hue', 'delta_sat', 'delta_lum']
		scenes = []
		num_scenes_accepted = 0
		bits = vid.Video.getBits(filename, ignore_errors=True)
		bit_depth = bits
		for sceneinfo in scenelist:
			# stop creating scenes if we've hit the maximum number for any scene
			if set.scenedetect_max_scenes_per_video >= 0  and len(scenes) >= set.scenedetect_max_scenes_per_video:
				break

			startframe = sceneinfo[0]
			endframe = sceneinfo[1]
			numframes = endframe - startframe + 1

			# Check that the scene has a minimum number of frames to be accepted
			if numframes < set.scenedetect_min_accept_length:
				continue

			# Check that the scene is not longer than the maximum acceptance length
			if set.scenedetect_max_accept_length >0:
				if numframes > set.scenedetect_max_accept_length:
					endframe = startframe + set.scenedetect_max_accept_length-1
					numframes = set.scenedetect_max_accept_length

			# define the scene
			scene = {}
			scene['video_name'] = tail
			scene['video_index'] = num_scenes_accepted
			scene['start_frame'] = startframe
			scene['start_framecode'] = sceneinfo[2]
			scene['end_frame'] = endframe
			scene['end_framecode'] = sceneinfo[3]  #may not be correct if endframe was adjusted above
			scene['num_frames'] = numframes
			scene['frame_width'] = framewidth
			scene['frame_height'] = frameheight
			scene['bit_depth'] = bit_depth
			scene['cropped'] = False
			scene['cropped_num_frames'] = 0,
			scene['cropped_frame_width'] = 0,
			scene['cropped_frame_height'] = 0,
			scene['cropped_path'] = ""
			scene['scene_detect_downscale'] = set.scenedetect_downscale
			scene['fps'] = fps
			scene['duplicated_frames'] = duplicated_frames
			scene['classification'] = 0  # placeholder value
			scene['diff'] = 0.0
			scene['avgdiff'] = 0.0
			scene['delta_hsv'] = 0.0
			scene['delta_hue'] = 0.0
			scene['delta_sat'] = 0.0
			scene['delta_lum'] = 0.0
			scene['max_delta_hsv'] = 0.0
			scene['max_delta_hsv_frame'] = 0.0
			scene['detection_alg'] = set.scenedetect_algorithm
			scene['tests'] = {}

			validscene = True
			if set.scenedetect_algorithm == 'histogram':
				avgdiff_total = 0
				diff_total = 0
				numdiff_frames = 0

				for frame in range(startframe, endframe):

					if frame != startframe:
						diff_total += histogram_metrics[frame][0]
						avgdiff_total += histogram_metrics[frame][1]

						numdiff_frames += 1

				scene['diff'] = diff_total / numdiff_frames
				scene['avgdiff'] = avgdiff_total / numdiff_frames

			else:
				# Get metrics for the scene.  These are delta hue, saturation, value for the frame from the previous frames
				dhsv_total = 0
				dhue_total = 0
				dsat_total = 0
				dlum_total = 0
				max_dhsv = 0
				max_dhsv_frame = -1

				numhsvframes = 0

				for frame in range(startframe, endframe):
					# don't use the first frame in the scene because the delta values will be large
					if histogram_metrics[frame][0] != None and frame != startframe:
						dhsv_total = dhsv_total + histogram_metrics[frame][0]
						dhue_total = dhue_total + histogram_metrics[frame][1]
						dsat_total = dsat_total + histogram_metrics[frame][2]
						dlum_total = dlum_total + histogram_metrics[frame][3]
						if histogram_metrics[frame][0] > max_dhsv:
							max_dhsv = histogram_metrics[frame][0]
							max_dhsv_frame = frame-startframe
						numhsvframes = numhsvframes + 1

						# with PySceneDetect, the first scene could have frames with a large threshold
						# add check to makes sure all frames in the scene (aside from the starting frame)
						# are within the threshold
						if metrics[0] > set.scenedetect_threshold:
							validscene = False
							break
				scene['delta_hsv'] = dhsv_total / numhsvframes
				scene['delta_hue'] = dhue_total / numhsvframes
				scene['delta_sat'] = dsat_total / numhsvframes
				scene['delta_lum'] = dlum_total / numhsvframes
				scene['max_delta_hsv'] = max_dhsv
				scene['max_delta_hsv_frame'] = max_dhsv_frame

			if not validscene:
				continue

			scenes.append(scene)
			num_scenes_accepted = num_scenes_accepted + 1

		return scenes, num_scenes_accepted, len(scenelist)

	def scenedetectresult(self, video, num_scenes_accepted, num_scenes_detected):
		if  num_scenes_detected == 0:
			logging.warning("WARNING - StageSceneDetect..video complete but no scenes detected: " + video + "(" + str(
				num_scenes_detected) + " scenes detected  " + str(num_scenes_accepted) + " scenes accepted)")
		elif num_scenes_accepted == 0:
			logging.warning("WARNING - StageSceneDetect..video complete but no scenes accepted: " + video + "(" + str(
				num_scenes_detected) + " scenes detected  " + str(num_scenes_accepted) + " scenes accepted)")
		else:
			logging.info(
				"StageSceneDetect..video complete: " + video+ "(" + str(num_scenes_detected) + " scenes detected  " + str(
					num_scenes_accepted) + " scenes accepted)")

	def scenedetect(self, filename, mutex, set, createdscenes):
		head, tail = os.path.split(filename)
		scenes, num_scenes_accepted, num_scenes_detected = self.detect(filename, set)

		mutex.acquire()
		for scene in scenes:
			createdscenes.append(scene)

		mutex.release()

		pass

	def ExecuteStage(self):
		logging.info("StageSceneDetect..executing stage")

		if settings.set.scenedetect_skip_stage:
			logging.info("StageSceneDetect..skipping stage")
			return

		if settings.set.scenedetect_min_accept_length > settings.set.scenedetect_max_accept_length:
			logging.warning("WARNING: scene detection min acceptance length: "+ str(settings.set.scenedetect_min_accept_length)+ " is not smaller than the maximum acceptance length: "+ str(settings.set.scenedetect_max_accept_length))
			return

		# Go through all of the filtered video files in the download folder
		filelist = []
		for filter in settings.set.scenedetect_video_filters:
			filelist += glob.glob(settings.set.absolute_videos_folder + os.sep + filter)

		if settings.set.multiprocess:
			processes = []
			mutex = mp.Lock()

			# In Windows processes are not forked as in Linux / Unix.Instead they are spawned, which means that anew
			# Python interpreter is started for each new multiprocessing.Process.This means that all global variables
			# are re-initialized and if you have somehow manipulated them along the way, this will not be seen by
			# the spawned processes.
			# https: // stackoverflow.com / questions / 49343907 / does - multiprocess - in -python - re - initialize - globals
			setcopy = copy.deepcopy(settings.set)

			manager = mp.Manager()
			scenelist = manager.list()

			for filename in filelist:
				head, tail = os.path.split(filename)
				logging.info("StageSceneDetect..processing video:" + tail)

				if database.db.containsVideo(tail):
					logging.warning("WARNING: Video already found in database:" + tail)
					if not settings.set.scenedetect_overwrite:
						continue

				p = mp.Process(target=self.scenedetect, args=(filename, mutex, setcopy, scenelist))
				processes.append(p)

			[p.start() for p in processes]
			[p.join() for p in processes]
			logging.info("StageSceneDetect..detection complete for all videos")

			# copy back
			settings.set = copy.deepcopy(setcopy)
		else:
			scenelist = []
			for filename in filelist:
				head, tail = os.path.split(filename)
				logging.info("StageSceneDetect..processing video:" + tail)

				if database.db.containsVideo(tail):
					logging.warning("Video already found in database:" + tail)
					if not settings.set.scenedetect_overwrite:
						continue

				scenes, num_scenes_accepted, num_scenes_detected = self.detect(filename, settings.set)
				for scene in scenes:
					scenelist.append(scene)

				self.scenedetectresult(tail, num_scenes_accepted, num_scenes_detected)

		sceneid = 0
		for scene in scenelist:

			if settings.set.scenedetect_max_scenes >= 0 and database.db.getNumScenes() >= settings.set.scenedetect_max_scenes:
				break

			scene['tests'] = {}
			database.db.add(scene, True)
			sceneid += 1
		logging.info("StageSceneDetect..# scenes detected:" + str(len(scenelist)))
		database.db.list()
		database.db.save()
		database.db.make_scene_sheet()
