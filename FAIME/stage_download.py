# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
from stage import Stage, StageEnum
import wget
import os.path
from os import path
from pathlib import Path

STAGE_NAME = 'StageDownload'

class StageDownload(Stage):

	def __init__(self):
		self.enum = StageEnum.DOWNLOAD
		pass

	def ExecuteStage(self):
		logging.info("StageDownload..executing stage")

		if settings.set.download_skip_stage:
			logging.info("StageDownload..skipping stage")
			return
		import youtube_dl
		# get a list of url videos from the string (strip off any trailing whitespace and delimit with cr)
		if settings.set.download_video_urls.find(',') != -1:
			list_of_video_urls = [str(item) for item in settings.set.download_video_urls.rstrip().split(',')]
		elif settings.set.download_video_urls.find('\n') != -1:
			list_of_video_urls = [str(item) for item in settings.set.download_video_urls.rstrip().split('\n')]
		elif settings.set.download_video_urls == '':
			list_of_video_urls = []
		else:
			list_of_video_urls  = [settings.set.download_video_urls.rstrip()]

		# get a list of vimeo videos from the string (strip off any trailing whitespace and delimit with cr)
		if settings.set.download_video_vimeos.find(',') != -1:
			list_of_vimeos = [str(item) for item in settings.set.download_video_vimeos.rstrip().split(',')]
		elif settings.set.download_video_vimeos.find('\n') != -1:
			list_of_vimeos = [str(item) for item in settings.set.download_video_vimeos.rstrip().split('\n')]
		elif settings.set.download_video_vimeos == '':
			list_of_vimeos = []
		else:
			list_of_vimeos = [settings.set.download_video_vimeos.rstrip()]

		# get a list of youtube videos from the string (strip off any trailing whitespace and delimit with cr)
		if settings.set.download_video_youtubes.find(',') != -1:
			list_of_youtubes = [str(item) for item in settings.set.download_video_youtubes.rstrip().split(',')]
		elif settings.set.download_video_youtubes.find('\n') != -1:
			list_of_youtubes = [str(item) for item in settings.set.download_video_youtubes.rstrip().split('\n')]
		elif settings.set.download_video_youtubes == '':
			list_of_youtubes = []
		else:
			list_of_youtubes = [settings.set.download_video_youtubes.rstrip()]

		#list_of_videos = open(settings.download_filelist,'r').read().split('\n')
		#from numpy import loadtxt
		#with open(settings.download_filelist) as f:
	    #		for line in f:
	    #			list_of_videos.append(line)

		# determine the folder and create it if it doesn't exist
		videofolder = os.path.abspath(settings.set.absolute_videos_folder)
		if not path.exists(videofolder):
			os.makedirs(videofolder)

		logging.info("StageDownload: num of video url files="+str(len(list_of_video_urls)))
		for video_name in list_of_video_urls:
			if len(video_name)==0:
				continue
			logging.info("StageDownload: video_name="+video_name)
			firstpos = video_name.rfind("/")
			lastpos = len(video_name)
			filename = video_name[firstpos+1:lastpos]

			# determine the download filename
			downloadpath = os.path.join(settings.set.absolute_videos_folder, filename)
			logging.info("StageDownload: file to be downloaded: " + downloadpath)

			filepath = Path(downloadpath)
			if not filepath.is_file() or settings.set.download_overwrite:
				# do the download
				logging.info("StageDownload: downloading video: " + video_name + " to: "+ downloadpath)
				wget.download(video_name, downloadpath)
			else:
				logging.warning("WARNING: StageDownload..video is already downloaded: "+ downloadpath)
			print("") # skip to beginning of next line

		logging.info("StageDownload: num of vimeos files="+str(len(list_of_vimeos)))
		for video_name in list_of_vimeos:
			if len(video_name) == 0:
				continue
			logging.info("StageDownload: video_name="+video_name)

			link_path = "https://vimeo.com/"
			tar_vid_input = link_path + video_name
			# print(tar_vid_input)
			info_dict = {"width": -1, "height": -1, "ext": "mp4", }

			downloadpath = os.path.join(videofolder, video_name +'.' + info_dict["ext"])
			filepath = Path(downloadpath)
			if not filepath.is_file() or settings.set.download_overwrite:
				# do the download
				logging.info("StageDownload: downloading video: " + video_name + " to: " + downloadpath)

				ydl = youtube_dl.YoutubeDL(
					{'format': 'bestvideo/best',
					 'outtmpl': os.path.join(videofolder, '%(id)s.%(ext)s'), })
				# download video from vimeo
				try:
					info_dict = ydl.extract_info(tar_vid_input, download=True)
				# we only need info_dict["ext"], info_dict["width"], info_dict["height"]
				except KeyboardInterrupt:
					print("KeyboardInterrupt!")
					exit()
				except:
					print("youtube_dl error:" + tar_vid_input)
					pass
				print("") # skip to beginning of next line
			else:
				logging.warning("WARNING: StageDownload..video is already downloaded: "+ downloadpath)

		logging.info("StageDownload: num of youtube files=" + str(len(list_of_youtubes)))
		for video_name in list_of_youtubes:
			if len(video_name) == 0:
				continue
			logging.info("StageDownload: video_name=" + video_name)

			link_path = "https://www.youtube.com/watch?v="
			tar_vid_input = link_path + video_name
			info_dict = {"width": -1, "height": -1, "ext": "mp4", }

			downloadpath = os.path.join(videofolder, video_name + '.' + info_dict["ext"])
			filepath = Path(downloadpath)
			if not filepath.is_file() or settings.set.download_overwrite:
				# do the download
				logging.info("StageDownload: downloading video: " + video_name + " to: " + downloadpath)
				# Specifed mp4 format first because .webm videos tend to lead to
				# empty frames being detected in later stages where individual frames
				# are processed
				ydl = youtube_dl.YoutubeDL(
					{'format': 'mp4/bestvideo/best',
					 'outtmpl': os.path.join(videofolder, '%(id)s.%(ext)s'), })
				# download video from vimeo
				try:
					info_dict = ydl.extract_info(tar_vid_input, download=True)
				# we only need info_dict["ext"], info_dict["width"], info_dict["height"]
				except KeyboardInterrupt:
					print("KeyboardInterrupt!")
					exit()
				except:
					print("youtube_dl error:" + tar_vid_input)
					pass
				print("")  # skip to beginning of next line
			else:
				logging.warning("WARNING: StageDownload..video is already downloaded: " + downloadpath)
		pass