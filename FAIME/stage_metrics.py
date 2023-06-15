# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import coloredlogs
from stage import Stage, StageEnum
import settings
import database
import os
import platform
import subprocess
import json
from pathlib import Path
import copy
import multiprocessing as mp
import glob
import math
import utils
from shutil import copyfile

STAGE_NAME = 'StageMetrics'

class StageMetrics(Stage):

	def __init__(self):
		self.enum = StageEnum.METRICS
		pass

	# MSSSIM code taken from:
	# http://places.csail.mit.edu/deepscene/small-projects/TRN-pytorch-pose/model_zoo/models/compression/msssim.py
	def _FSpecialGauss(self, size, sigma):
		import numpy as np
		"""Function to mimic the 'fspecial' gaussian MATLAB function."""
		radius = size // 2
		offset = 0.0
		start, stop = -radius, radius + 1
		if size % 2 == 0:
			offset = 0.5
			stop -= 1
		x, y = np.mgrid[offset + start:stop, offset + start:stop]
		assert len(x) == size
		g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
		return g / g.sum()

	def _SSIMForMultiScale(self, img1, img2, max_val=255, filter_size=11,
						   filter_sigma=1.5, k1=0.01, k2=0.03):
		from scipy import signal
		import numpy as np
		"""Return the Structural Similarity Map between `img1` and `img2`.

        This function attempts to match the functionality of ssim_index_new.m by
        Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

        Arguments:
          img1: Numpy array holding the first RGB image batch.
          img2: Numpy array holding the second RGB image batch.
          max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
          filter_size: Size of blur kernel to use (will be reduced for small images).
          filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
          k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
          k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).

        Returns:
          Pair containing the mean SSIM and contrast sensitivity between `img1` and
          `img2`.

        Raises:
          RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
        """
		if img1.shape != img2.shape:
			raise RuntimeError('Input images must have the same shape (%s vs. %s).',
							   img1.shape, img2.shape)
		if img1.ndim != 4:
			raise RuntimeError('Input images must have four dimensions, not %d',
							   img1.ndim)

		img1 = img1.astype(np.float64)
		img2 = img2.astype(np.float64)
		_, height, width, _ = img1.shape

		# Filter size can't be larger than height or width of images.
		size = min(filter_size, height, width)

		# Scale down sigma if a smaller filter size is used.
		sigma = size * filter_sigma / filter_size if filter_size else 0

		if filter_size:
			window = np.reshape(self._FSpecialGauss(size, sigma), (1, size, size, 1))
			mu1 = signal.fftconvolve(img1, window, mode='valid')
			mu2 = signal.fftconvolve(img2, window, mode='valid')
			sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
			sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
			sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
		else:
			# Empty blur kernel so no need to convolve.
			mu1, mu2 = img1, img2
			sigma11 = img1 * img1
			sigma22 = img2 * img2
			sigma12 = img1 * img2

		mu11 = mu1 * mu1
		mu22 = mu2 * mu2
		mu12 = mu1 * mu2
		sigma11 -= mu11
		sigma22 -= mu22
		sigma12 -= mu12

		# Calculate intermediate values used by both ssim and cs_map.
		c1 = (k1 * max_val) ** 2
		c2 = (k2 * max_val) ** 2
		v1 = 2.0 * sigma12 + c2
		v2 = sigma11 + sigma22 + c2
		ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
		cs = np.mean(v1 / v2)
		return ssim, cs

	def MultiScaleSSIM(self, img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
		from scipy.ndimage.filters import convolve
		import numpy as np
		"""Return the MS-SSIM score between `img1` and `img2`.

		This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
		Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
		similarity for image quality assessment" (2003).
		Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

		Author's MATLAB implementation:
		http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

		Arguments:
		  img1: Numpy array holding the first RGB image batch.
		  img2: Numpy array holding the second RGB image batch.
		  max_val: the dynamic range of the images (i.e., the difference between the
		    maximum the and minimum allowed values).
		  filter_size: Size of blur kernel to use (will be reduced for small images).
		  filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
		    for small images).
		  k1: Constant used to maintain stability in the SSIM calculation (0.01 in
		    the original paper).
		  k2: Constant used to maintain stability in the SSIM calculation (0.03 in
		    the original paper).
		  weights: List of weights for each level; if none, use five levels and the
		    weights from the original paper.

		Returns:
		  MS-SSIM score between `img1` and `img2`.

		Raises:
		  RuntimeError: If input images don't have the same shape or don't have four
		    dimensions: [batch_size, height, width, depth].
		"""
		if img1.shape != img2.shape:
			raise RuntimeError('Input images must have the same shape (%s vs. %s).',
							   img1.shape, img2.shape)
		if img1.ndim != 4:
			raise RuntimeError('Input images must have four dimensions, not %d',
							   img1.ndim)

		# Note: default weights don't sum to 1.0 but do match the paper / matlab code.
		weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
		weights = np.array(weights if weights else
						   [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
		levels = weights.size
		downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
		im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
		mssim = np.array([])
		mcs = np.array([])
		# xrange used in orginal version (Python 2.x only)
		for _ in range(levels):
			ssim, cs = self._SSIMForMultiScale(
				im1, im2, max_val=max_val, filter_size=filter_size,
				filter_sigma=filter_sigma, k1=k1, k2=k2)
			mssim = np.append(mssim, ssim)
			mcs = np.append(mcs, cs)
			filtered = [convolve(im, downsample_filter, mode='reflect')
						for im in [im1, im2]]
			im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
		return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
				(mssim[levels - 1] ** weights[levels - 1]))

	def CopyAlexnetModel(self, setcopy):
		alexpathorg = os.path.join(setcopy.absolute_models_folder, "alexnet-owt-4df8aa71.pth")
		if os.path.exists(alexpathorg):
			torch_home = os.path.expanduser(
				os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
			hub_dir = os.path.join(torch_home, 'hub')
			model_dir = os.path.join(hub_dir, 'checkpoints')
			Path(model_dir).mkdir(parents=True, exist_ok=True)

			alexpath = os.path.join(model_dir, "alexnet-owt-4df8aa71.pth")
			if not os.path.exists(alexpath):
				copyfile(alexpathorg, alexpath)

	def CollectMetricsForScene(self, sceneid, setcopy, scene, test_name, HR_folder, test_folder):
		import cv2
		import numpy as np
		import torch
		use_cuda = torch.cuda.is_available()

		# Configure Logging
		logger = logging.getLogger(__name__)
		coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

		hrfiles = utils.GetFiles(HR_folder)
		if len(hrfiles)== 0:
			logging.warning("StageMetrics.. no HR files found in: "+ HR_folder)
			return

		testfiles = utils.GetFiles(test_folder)
		if len(testfiles) == 0:
			logging.warning("StageMetrics.. no test files found in: " + test_folder)
			return

		# get the path to the mestrics file
		metrics_path = os.path.join(test_folder, "metrics.json")

		# initialize the metrics data
		metrics_data = {}

		# set skip flags to avoid calculating existing metrics if not overwriting them
		skip_psnr = False
		skip_ssim = False
		skip_msssim = False
		skip_vmaf = False
		skip_gmaf = False
		skip_lpips = False
		skip_haarpsi = False

		if os.path.exists(metrics_path):
			with open(metrics_path, 'r') as f:
				metrics_data = json.load(f)
				if 'psnr_yuv' in metrics_data and not setcopy.metrics_overwrite:
					skip_psnr= True
				if 'ssim_yuv' in metrics_data and not setcopy.metrics_overwrite:
					skip_ssim = True
				if 'msssim' in metrics_data and not setcopy.metrics_overwrite:
					skip_msssim = True
				if 'vmaf' in metrics_data and not setcopy.metrics_overwrite:
					skip_vmaf = True
				if 'gmaf' in metrics_data and not setcopy.metrics_overwrite:
					skip_gmaf = True
				if 'lpips' in metrics_data and not setcopy.metrics_overwrite:
					skip_lpips = True
				if 'haarpsi' in metrics_data and not setcopy.metrics_overwrite:
					skip_haarpsi = True

		# quick exit if there are no metrics to calculate...
		if skip_psnr and skip_ssim and skip_msssim and skip_vmaf and skip_gmaf and skip_lpips and skip_haarpsi:
			logging.warning("StageMetrics.. no metrics needed to calculate for scene " + str(sceneid) + " test:" + test_name)
			return

		# initialize per metric lists of frame scores
		psnr_yuvs = []
		psnr_ys = []
		psnr_us = []
		psnr_vs = []
		ssim_yuvs = []
		ssim_ys = []
		ssim_us = []
		ssim_vs = []
		msssims = []
		vmafs = []
		gmafs = []
		lpipss = []
		haarpsis = []

		combinedyuv1 = []
		combinedyuv2 = []
		width = 0
		height = 0

		psnr_yuv_score = 0.0
		psnr_y_score = 0.0
		psnr_u_score = 0.0
		psnr_v_score = 0.0

		ssim_yuv_score = 0.0
		ssim_y_score = 0.0
		ssim_u_score = 0.0
		ssim_v_score = 0.0

		msssim_score = 0.0
		lpips_score = 0.0
		haarpsi_score = 0.0

		# loop over every hr file
		numframes = len(hrfiles)
		for index, hr_path in enumerate(hrfiles):

			# report progress
			if index % 10 == 0:
				logging.info("StageMetrics: "+ str(index) + " of "+ str(numframes) + " calculated for scene " + str(sceneid) + " test:" + test_name)

			if index >= len(testfiles):
				logging.warning("WARNING: Inferred test file does not exist for HR file: " + hr_path + " ending metric collection")
				break

			test_path = testfiles[index]

			if not os.path.exists(hr_path):
				logging.warning("WARNING: Original HR file does not exist: " + hr_path)
				continue

			if not os.path.exists(test_path):
				logging.warning("WARNING: Inferred VSR file does not exist: " + test_path)
				continue

			# Read the 2 files to compare
			img1 = cv2.imread(hr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
			img2 = cv2.imread(test_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

			# do cropping to mke sure the images are the same size
			if img1.shape[:2] != img2.shape[:2]:
				if (img2.shape[0] > img1.shape[0] and img2.shape[1] >= img1.shape[1]) or \
						(img2.shape[1] > img1.shape[1] and img2.shape[0] >= img1.shape[0]):
					# test image is larger than the original (in 1 or both dimensions.
					# First log the warning on the fist image
					if index == 0:
						logging.warning(
							"WARNING: Test image " + test_path + "("+ str(img2.shape[1]) + "x" + str(img2.shape[0]) +
							") is larger than HR image " + hr_path + "(" + str(img1.shape[1]) + "x" + str(img1.shape[0]) +")")

					# Crop the test image to match the same size as the hr path
					img2 = img2[0:img1.shape[0], 0:img1.shape[1]]

				else:
					logging.warning("WARNING: Test and HR files have different shapes. Unable to compute metrics: original="+hr_path+" test="+test_path )
					continue

			# crop the frame images to a centered 32x32 multiple (remove any border artifacts)
			if setcopy.metrics_border_crop:
				# cropping logic is taken from metrics.py in the TecoGAN project
				ori_h = img1.shape[0]
				ori_w = img1.shape[1]

				h = (ori_h // 32) * 32
				w = (ori_w // 32) * 32

				while (h > ori_h - 16):
					h = h - 32
				while (w > ori_w - 16):
					w = w - 32

				y = (ori_h - h) // 2
				x = (ori_w - w) // 2
				img1 = img1[y:y + h, x:x + w]
				img2 = img2[y:y + h, x:x + w]

			# LPIPS
			if setcopy.metrics_lpips and not skip_lpips:
				logging.info("processing lpips")
				import lpips
				loss_fn = lpips.LPIPS(net='alex')
				lpips_img1 = lpips.im2tensor(img1) # RGB image from [-1,1]
				lpips_img2 = lpips.im2tensor(img2)
				if not setcopy.test_cpu_only and use_cuda:
					img1 = img1.cuda()
					img2 = img2.cuda()
				# Compute distance
				dist = loss_fn.forward(lpips_img1,lpips_img2)
				dist_float =float(dist)
				lpipss.append(dist_float)

			# PSNR
			if setcopy.metrics_psnr and not skip_psnr:
				logging.info("processing psnr")
				from skimage.metrics import peak_signal_noise_ratio as psnr
				# convert the BGR frame data to YUV
				yuvimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
				yuvimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

				# psnr_result = cv2.PSNR(img1, img2)
				psnr_calc = psnr(yuvimg1, yuvimg2)
				if math.isinf(psnr_calc):
					logging.warning("WARNING: Infinite PSNR yuv calculated: " + hr_path + " and" + test_path)
				else:
					psnr_yuvs.append(psnr_calc)

				y1, u1, v1 = cv2.split(yuvimg1)
				y2, u2, v2 = cv2.split(yuvimg2)
				psnr_calc = psnr(y1, y2)
				if math.isinf(psnr_calc):
					logging.warning("WARNING: Infinite PSNR y calculated: " + hr_path + " and" + test_path)
				else:
					psnr_ys.append(psnr_calc)
				psnr_calc = psnr(u1,u2)
				if math.isinf(psnr_calc):
					logging.warning("WARNING: Infinite PSNR u calculated: " + hr_path + " and" + test_path)
				else:
					psnr_us.append(psnr_calc)

				psnr_calc = psnr(v1,v2)
				if math.isinf(psnr_calc):
					logging.warning("WARNING: Infinite PSNR v calculated: " + hr_path + " and" + test_path)
				else:
					psnr_vs.append(psnr_calc)
			else:
				psnr_yuvs.append(0.0)
				psnr_ys.append(0.0)
				psnr_us.append(0.0)
				psnr_vs.append(0.0)

			# SSIM
			if setcopy.metrics_ssim and not skip_ssim:
				logging.info("processing ssim")
				from skimage.metrics import structural_similarity as ssim
				# convert the BGR frame data to YUV
				yuvimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
				yuvimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

				ssim_result = ssim(yuvimg1, yuvimg2, multichannel=True)
				ssim_yuvs.append(ssim_result)

				y1, u1, v1 = cv2.split(yuvimg1)
				y2, u2, v2 = cv2.split(yuvimg2)
				ssim_ys.append(ssim(y1, y2, multichannel=False))
				ssim_us.append(ssim(u1, u2, multichannel=False))
				ssim_vs.append(ssim(v1, v2, multichannel=False))

			else:
				ssim_yuvs.append(0.0)
				ssim_ys.append(0.0)
				ssim_us.append(0.0)
				ssim_vs.append(0.0)

			# MSSSIM
			if setcopy.metrics_msssim and not skip_msssim:
				logging.info("processing mssim")
				import tensorflow.compat.v1 as tf
				tf.disable_v2_behavior()
				with tf.io.gfile.GFile(hr_path, 'rb') as image_file:
					img1_str = image_file.read()
				with tf.io.gfile.GFile(test_path, 'rb') as image_file:
					img2_str = image_file.read()

				input_img = tf.placeholder(tf.string)
				decoded_image = tf.expand_dims(tf.image.decode_png(input_img, channels=3), 0)

				with tf.Session() as sess:
					msimg1 = sess.run(decoded_image, feed_dict={input_img: img1_str})
					msimg2 = sess.run(decoded_image, feed_dict={input_img: img2_str})

				# MultiScaleSSIM expects images to be in RGB format
				msssim_result = self.MultiScaleSSIM(msimg1, msimg2)
				msssims.append(msssim_result)
			else:
				msssims.append(0.0)

			if (setcopy.metrics_vmaf and not skip_vmaf) or (setcopy.metrics_gmaf and not skip_gmaf):
				logging.info("processing vmaf and gmaf")

				if not setcopy.metrics_single_frame:
					# process scene as a video sequence
					if not scene['same_size']:
						logging.warning("WARNING: Unable to calculate VMAF for scene as a sequence because frames are not the same size: " +
										scene['folder_name'])
						vmafs.append(0.0)
					else:
						# From Raul 3/2021: please use the latest version of VMAF (released Dec 2020) and set the "No Gain" NEG
						# parameter to 1.0 for both VIF and ADM.
						# VMAF is designed to test video at 1080p.  If you are testing another video resolution, you MUST up
						# sample the video to 1080p using the default model.  If you are going to test 4K, there is a
						# special mode that you need to select which loads a different model (see
						# github.com / Netfilx / vmaf / tree / model / vmaf_4k_v0.6.1pk1).

						frameheight, framewidth = img1.shape[:2]
						if frameheight < 1080:
							# scale the frames to 1080.  Make sure the width is a multiple of 2 so the conversion to I420 works
							newwidth = int(framewidth * float(1080)/float(frameheight))
							if newwidth % 2 != 0:
								newwidth = newwidth + 1
							resizedimg1 = cv2.resize(img1, dsize=(newwidth, 1080), interpolation=cv2.INTER_CUBIC)
							resizedimg2 = cv2.resize(img2, dsize=(newwidth, 1080), interpolation=cv2.INTER_CUBIC)
						else:
							resizedimg1 = img1
							resizedimg2 = img2

						# convert the BGR frame data to YUV420
						resizedimg1=resizedimg1.astype(np.uint8)
						resizedimg2=resizedimg2.astype(np.uint8)
						frame_yuv1 = cv2.cvtColor(resizedimg1, cv2.COLOR_BGR2YUV_I420)
						frame_yuv2 = cv2.cvtColor(resizedimg2, cv2.COLOR_BGR2YUV_I420)

						yuv1_frame_data = np.asarray(frame_yuv1)
						yuv2_frame_data = np.asarray(frame_yuv2)

						# append the yuv data together so that a yuv file can be saved for vmaf/gmaf processing
						if hr_path == hrfiles[0]:
							combinedyuv1 = yuv1_frame_data
							combinedyuv2 = yuv2_frame_data
							height, width = resizedimg1.shape[:2]
						else:
							combinedyuv1 = np.append(combinedyuv1, yuv1_frame_data)
							combinedyuv2 = np.append(combinedyuv2, yuv2_frame_data)
				else:
					# Process the scene as a set of independent frames
					# convert the BGR frame data to YUV420
					frame_yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV_I420)
					frame_yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV_I420)

					height, width = img1.shape[:2]

					# get file paths to the 2 yuv files we'll need for vmaf calculation
					reference_file_path = os.path.join(test_folder, "img_yuv1.yuv")
					distorted_file_path = os.path.join(test_folder, "img_yuv2.yuv")

					# save the yuv data to these files
					frame_yuv1.astype('uint8').tofile(reference_file_path)
					frame_yuv2.astype('uint8').tofile(distorted_file_path)

					# Use vmaf model v0.6.1 for 4k sized frames
					model_str = ''
					if width > 3500:
						model_str = '--model version=vmaf_v0.6.1 '

					# get file path the to json file which will contain the vmaf results
					vmaf_json_path = os.path.join(test_folder, "vmaf_report.json")

					if setcopy.metrics_vmaf_preload:
						cmd0 = "LD_PRELOAD='./FAIME/vmaf/libm.so.6' ./FAIME/vmaf/vmaf " + model_str + "--feature vif=vif_enhn_gain_limit=1.0  adm=adm_enhn_gain_limit=1.0 -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
								str(width) + " -h " + str(height) + " -b 8 --json -q -o " + vmaf_json_path
					else:
						cmd0 = "./FAIME/vmaf/vmaf " + model_str + "--feature vif=vif_enhn_gain_limit=1.0  adm=adm_enhn_gain_limit=1.0 -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
								str(width) + " -h " + str(height) + " -b 8 --json -q -o " + vmaf_json_path

					# run the vmaf program.  Override stdout to avoid seeing any unrelated information from vmaf
					FNULL = open(os.devnull, 'w')

					logging.info("INFO: vmaf cmd0 = "+ cmd0)
					subprocess.call(cmd0, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
					FNULL.close()

					# see if the vmaf json file was created
					vmafjsonpath = Path(vmaf_json_path)
					if vmafjsonpath.is_file():
						# oepn the file and process the metrics (mean value is the only one of interest)
						with open(vmaf_json_path, 'r') as f:
							jsondata = json.load(f)
							frames = jsondata['frames']
							metrics = frames[0]['metrics']
							vmafs.append(metrics["vmaf"])

					os.remove(reference_file_path)
					os.remove(distorted_file_path)

			haarpsi_score = 0.0
			if setcopy.metrics_haarpsi and not skip_haarpsi:
				logging.info("processing haarpsi")
				import haarpsi.haarPsi as Haarpsi
				haarpsis.append(Haarpsi.haar_psi(img1,img2)[0])
			else:
				haarpsis.append(haarpsi_score)


		# calculate psnr for the scene (average of frame psnrs)
		if setcopy.metrics_psnr and len(psnr_yuvs) > 0:
			psnr_yuv_score = sum(psnr_yuvs) / len(psnr_yuvs)
			psnr_y_score = sum(psnr_ys) / len(psnr_ys)
			psnr_u_score = sum(psnr_us) / len(psnr_us)
			psnr_v_score = sum(psnr_vs) / len(psnr_vs)
		if setcopy.metrics_lpips and len(lpipss) > 0:
			lpips_score = sum(lpipss)/ len(lpipss)
		# calculate ssim for the scene (average of frame ssims)
		if setcopy.metrics_ssim and len(ssim_yuvs)>0:
			ssim_yuv_score = sum(ssim_yuvs)/ len(ssim_yuvs)
			ssim_y_score = sum(ssim_ys)/ len(ssim_ys)
			ssim_u_score = sum(ssim_us)/ len(ssim_us)
			ssim_v_score = sum(ssim_vs)/ len(ssim_vs)

		# calculate msssim for the scene (average of frame msssims)
		if setcopy.metrics_msssim and len(msssims)>0:
			msssim_score = sum(msssims)/ len(msssims)

		# calculate haarpsi for the scene (average of frame haarpsi)
		if setcopy.metrics_haarpsi and len(haarpsis) > 0:
			haarpsi_score = sum(haarpsis) / len(haarpsis)

		# calculate vmaf by processing the yuv frame data
		vmaf_adm2_score = 0.0
		vmaf_adm_scale0_score = 0.0
		vmaf_adm_scale1_score = 0.0
		vmaf_adm_scale2_score = 0.0
		vmaf_adm_scale3_score = 0.0
		vmaf_adm_motion2_score = 0.0
		vmaf_adm_motion_score = 0.0
		vmaf_vif_scale0_score = 0.0
		vmaf_vif_scale1_score = 0.0
		vmaf_vif_scale2_score = 0.0
		vmaf_vif_scale3_score = 0.0
		vmaf_score = 0.0

		if setcopy.metrics_vmaf and not skip_vmaf:
			logging.info("processing vmaf scores")
			if not setcopy.metrics_single_frame:
				if len(combinedyuv1) != 0 and len(combinedyuv2) != 0 and len(combinedyuv1) == len(combinedyuv2):

					# get file paths to the 2 yuv files we'll need for vmaf calculation
					reference_file_path = os.path.join(test_folder,"img_yuv1.yuv")
					distorted_file_path = os.path.join(test_folder,"img_yuv2.yuv")

					# save the yuv data to these files
					combinedyuv1.astype('uint8').tofile(reference_file_path)
					combinedyuv2.astype('uint8').tofile(distorted_file_path)

					# get file path the to json file which will contain the vmaf results
					vmaf_json_path = os.path.join(test_folder, "vmaf_report.json")

					# run the vmaf program.  Override stdout to avoid seeing any unrelated information from vmaf
					FNULL = open(os.devnull, 'w')

					# Use vmaf model v0.6.1 for 4k sized frames
					model_str = ''
					if width > 3500:
						model_str = '--model version=vmaf_v0.6.1 '

					if setcopy.metrics_vmaf_preload:
						cmd0 = "LD_PRELOAD='./FAIME/vmaf/libm.so.6' ./FAIME/vmaf/vmaf " + model_str + "--feature vif=vif_enhn_gain_limit=1.0  adm=adm_enhn_gain_limit=1.0 -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
								str(width) + " -h " + str(height) + " -b 8 --json -q -o " + vmaf_json_path
					else:
						cmd0 = "./FAIME/vmaf/vmaf " + model_str + "--feature vif=vif_enhn_gain_limit=1.0  adm=adm_enhn_gain_limit=1.0 -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
								str(width) + " -h " + str(height) + " -b 8 --json -q -o " + vmaf_json_path

					logging.info("INFO: vmaf cmd0 = "+ cmd0)
					subprocess.call(cmd0, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
					FNULL.close()

					# see if the vmaf json file was created
					vmafjsonpath = Path(vmaf_json_path)
					if vmafjsonpath.is_file():
						# oepn the file and process the metrics (mean value is the only one of interest)
						with open(vmaf_json_path, 'r') as f:
							jsondata = json.load(f)
							pooled = jsondata['pooled_metrics']
							vmaf_adm2_score = pooled[0]['pooling_methods']['mean']
							vmaf_adm_scale0_score = pooled[1]['pooling_methods']['mean']
							vmaf_adm_scale1_score = pooled[2]['pooling_methods']['mean']
							vmaf_adm_scale2_score = pooled[3]['pooling_methods']['mean']
							vmaf_adm_scale3_score = pooled[4]['pooling_methods']['mean']
							vmaf_adm_motion2_score = pooled[5]['pooling_methods']['mean']
							vmaf_adm_motion_score = pooled[6]['pooling_methods']['mean']
							vmaf_vif_scale0_score = pooled[7]['pooling_methods']['mean']
							vmaf_vif_scale1_score = pooled[8]['pooling_methods']['mean']
							vmaf_vif_scale2_score = pooled[9]['pooling_methods']['mean']
							vmaf_vif_scale3_score = pooled[10]['pooling_methods']['mean']
							vmaf_score = pooled[11]['pooling_methods']['mean']

							# get the per frame vmaf scores:
							frames = jsondata['frames']
							for frame in frames:
								metrics = frame['metrics']
								vmafs.append(metrics["vmaf"])

						# don't delete the vmaf json report file
						# os.remove(vmaf_json_path)
					else:
						# no vmaf json file was created
						logging.warning("WARNING: Unable to collect VMAF data for yuvs: " + reference_file_path + " and " + distorted_file_path)

					# delete the yuv files now that vmaf is calculated
					if setcopy.metrics_vmaf_delete_yuvs:
						os.remove(reference_file_path)
						os.remove(distorted_file_path)
			else:
				# calculate an average vmaf_score for the scene
				for score in vmafs:
					vmaf_score += score
				if len(vmafs):
					vmaf_score = vmaf_score/len(vmafs)

		# calculate gmaf by processing the yuv frame data
		gmaf_adm2_score = 0.0
		gmaf_adm_scale0_score = 0.0
		gmaf_adm_scale1_score = 0.0
		gmaf_adm_scale2_score = 0.0
		gmaf_adm_scale3_score = 0.0
		gmaf_adm_motion2_score = 0.0
		gmaf_adm_motion_score = 0.0
		gmaf_vif_scale0_score = 0.0
		gmaf_vif_scale1_score = 0.0
		gmaf_vif_scale2_score = 0.0
		gmaf_vif_scale3_score = 0.0
		gmaf_score = 0.0

		if setcopy.metrics_gmaf and not skip_gmaf:
			logging.info("processing gmaf scores")
			if len(combinedyuv1) != 0 and len(combinedyuv2) != 0 and len(combinedyuv1) == len(combinedyuv2):

				# get file paths to the 2 yuv files we'll need for gmaf calculation
				reference_file_path = os.path.join(test_folder,"img_yuv1.yuv")
				distorted_file_path = os.path.join(test_folder,"img_yuv2.yuv")

				logging.info("processing gmaf scores..reference_file_path="+reference_file_path)
				logging.info("processing gmaf scores..distorted_file_path="+distorted_file_path)

				# save the yuv data to these files
				combinedyuv1.astype('uint8').tofile(reference_file_path)
				combinedyuv2.astype('uint8').tofile(distorted_file_path)

				# get file path the to json file which will contain the gmaf results
				gmaf_json_path = os.path.join(test_folder, "gmaf_report.json")

				# run the vmaf program.  Override stdout to avoid seeing any unrelated information from vmaf
				FNULL = open(os.devnull, 'w')

				# Use the model string
				model_str = '--model path=FAIME/gmaf/gmaf_v0.30.json'

				#if setcopy.metrics_gmaf_preload:
				#	cmd0 = "LD_PRELOAD='./vgaf/libm.so.6' ./gmaf/gmaf " + model_str + " -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
				#		   str(width) + " -h " + str(height) + " -b 8 --json -q -o " + gmaf_json_path
				#else:
				cmd0 = "./FAIME/gmaf/vmaf " + model_str + " -r " + reference_file_path + " -d " + distorted_file_path + " -p 420 -w " + \
							str(width) + " -h " + str(height) + " -b 8 --json -q -o " + gmaf_json_path

				logging.info("INFO: vgaf cmd0 = "+ cmd0)
				subprocess.call(cmd0, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
				FNULL.close()

				# see if the gmaf json file was created
				gmafjsonpath = Path(gmaf_json_path)
				if gmafjsonpath.is_file():
					# open the file and process the metrics (mean value is the only one of interest)
					with open(gmaf_json_path, 'r') as f:
						jsondata = json.load(f)
						pooled = jsondata['pooled_metrics']
						gmaf_adm2_score = pooled['adm2']['mean']
						gmaf_adm_scale0_score = pooled['adm_scale0']['mean']
						gmaf_adm_scale1_score = pooled['adm_scale1']['mean']
						gmaf_adm_scale2_score = pooled['adm_scale2']['mean']
						gmaf_adm_scale3_score = pooled['adm_scale3']['mean']
						gmaf_adm_motion2_score = pooled['motion2']['mean']
						gmaf_adm_motion_score = pooled['motion']['mean']
						gmaf_vif_scale0_score = pooled['vif_scale0']['mean']
						gmaf_vif_scale1_score = pooled['vif_scale1']['mean']
						gmaf_vif_scale2_score = pooled['vif_scale2']['mean']
						gmaf_vif_scale3_score = pooled['vif_scale3']['mean']
						gmaf_score = pooled['vmaf']['mean']

						# get the per frame vmaf scores:
						frames = jsondata['frames']
						for frame in frames:
							metrics = frame['metrics']
							gmafs.append(metrics["vmaf"])

					# don't delete the gmaf json report file
					# os.remove(gmaf_json_path)
				else:
					# no gmaf json file was created
					logging.warning("WARNING: Unable to collect GMAF data for yuvs: " + reference_file_path + " and " + distorted_file_path)

				# delete the yuv files now that gmaf is calculated
				if setcopy.metrics_gmaf_delete_yuvs:
					os.remove(reference_file_path)
					os.remove(distorted_file_path)

		logging.info("processing metrics_data")
		# update the metrics data depending on which metrics were calculated
		if setcopy.metrics_psnr and not skip_psnr:
			metrics_data.update({
				'psnr_yuv'         : psnr_yuv_score,
                'psnr_y'           : psnr_y_score,
				'psnr_u'           : psnr_u_score,
				'psnr_v'           : psnr_v_score,
				'psnr_yuvs'        : psnr_yuvs,
				'psnr_ys'          : psnr_ys,
				'psnr_us'          : psnr_us,
				'psnr_vs'          : psnr_vs})

		if setcopy.metrics_ssim and not skip_ssim:
			metrics_data.update({
				'ssim_yuv'    	: ssim_yuv_score,
				'ssim_y'		: ssim_y_score,
				'ssim_u'		: ssim_u_score,
				'ssim_v'		: ssim_v_score,
				'ssim_yuvs'     : ssim_yuvs,
				'ssim_ys'		: ssim_ys,
				'ssim_us'		: ssim_us,
				'ssim_vs'		: ssim_vs})

		if setcopy.metrics_msssim and not skip_msssim:
			metrics_data.update({
				'msssim'    : msssim_score,
                'msssims': msssims})

		if setcopy.metrics_vmaf and not skip_vmaf:
			metrics_data.update({
				'vmaf_adm2': vmaf_adm2_score,
				'vmaf_adm_scale0': vmaf_adm_scale0_score,
				'vmaf_adm_scale1': vmaf_adm_scale1_score,
				'vmaf_adm_scale2': vmaf_adm_scale2_score,
				'vmaf_adm_scale3': vmaf_adm_scale3_score,
				'vmaf_adm_motion2': vmaf_adm_motion2_score,
				'vmaf_adm_motion': vmaf_adm_motion_score,
				'vmaf_vif_scale0': vmaf_vif_scale0_score,
				'vmaf_vif_scale1': vmaf_vif_scale1_score,
				'vmaf_vif_scale2': vmaf_vif_scale2_score,
				'vmaf_vif_scale3': vmaf_vif_scale3_score,
				'vmaf'           : vmaf_score,
                'vmafs'          : vmafs})

		if setcopy.metrics_gmaf and not skip_gmaf:
			metrics_data.update({
				'gmaf_adm2': gmaf_adm2_score,
				'gmaf_adm_scale0': gmaf_adm_scale0_score,
				'gmaf_adm_scale1': gmaf_adm_scale1_score,
				'gmaf_adm_scale2': gmaf_adm_scale2_score,
				'gmaf_adm_scale3': gmaf_adm_scale3_score,
				'gmaf_adm_motion2': gmaf_adm_motion2_score,
				'gmaf_adm_motion': gmaf_adm_motion_score,
				'gmaf_vif_scale0': gmaf_vif_scale0_score,
				'gmaf_vif_scale1': gmaf_vif_scale1_score,
				'gmaf_vif_scale2': gmaf_vif_scale2_score,
				'gmaf_vif_scale3': gmaf_vif_scale3_score,
				'gmaf'           : gmaf_score,
                'gmafs'          : gmafs})

		if setcopy.metrics_lpips and not skip_lpips:
			metrics_data.update({
				'lpips'   : lpips_score,
				'lpipss'  : lpipss})

		if setcopy.metrics_haarpsi and not skip_haarpsi:
			metrics_data.update({
				'haarpsi'    : haarpsi_score,
                'haarpsis'   : haarpsis})

		logging.info("processing metrics_data2")

		# save results to metrics.json
		with open(metrics_path, 'w') as f:
			json.dump(metrics_data, f, indent=2, separators=(',', ':'))

		logging.info("StageMetrics.. HR folder:" + HR_folder + " test folder: "+ test_folder + " # of frames: " + str(numframes) +
					 " PSNR="  + "{0:8.4f}".format(psnr_yuv_score) +
					 " SSIM="  + "{0:8.4f}".format(ssim_yuv_score) +
					 " MSSSIM="+ "{0:8.4f}".format(msssim_score) +
					 " VMAF="  + "{0:8.4f}".format(vmaf_score) +
					 " GMAF="  + "{0:8.4f}".format(gmaf_score) +
					 " LPIPS=" + "{0:8.4f}".format(lpips_score) +
					 " HAARPSI="+"{0:8.4f}".format(haarpsi_score))
		logging.info("processing done for scene")

	def ExecuteStage(self):
		logging.info("StageMetrics.executing stage")

		if settings.set.metrics_skip_stage:
			logging.info("StagemMetrics..skipping stage")
			return

		sceneindices = database.db.getSceneIndices()

		logging.info("StageMetrics.." + str(len(sceneindices) )+ " scene(s) to gather metrics on")

		# location where the scene folders of inferred frames are saved
		base_test_folder = settings.set.absolute_test_folder
		if not os.path.exists(base_test_folder):
			logging.warning("WARNING: base test folder does not exist: " + base_test_folder)
			return

		# location of test folders
		test_folder = os.path.join(base_test_folder, settings.set.metrics_test_name)
		if not os.path.exists(test_folder):
			logging.warning("WARNING: test folder does not exist for test: " + test_folder)
			return

		lastsceneindex = sceneindices[-1]

		if settings.set.multiprocess:
			processes = []
			setcopy = copy.deepcopy(settings.set)

			for sceneindex in sceneindices:
				scene = database.db.getScene(sceneindex)

				if settings.set.metrics_test_name != "" and settings.set.metrics_test_name not in scene['tests']:
					logging.info("StageMetrics...skipping scene " + str(scene['scene_index']) + " no test found with name: " + settings.set.metrics_test_name)
					continue

				if scene['cropped']:
					hr_folder = scene['cropped_path']
				else:
					hr_folder = scene['hr_path']

				if not os.path.exists(hr_folder):
					logging.warning("StageMetrics...WARNING HR folder does not exist for scene: " + hr_folder)
					continue

				lasttestname = list(scene['tests'])[-1]
				for test_name in scene['tests']:

					if settings.set.metrics_test_name != "" and test_name != settings.set.metrics_test_name:
						continue

					if 'test_path' not in scene['tests'][test_name]:
						logging.info("StageMetrics...skipping scene " + str(scene['scene_index']) + " because test path not found")
						continue

					logging.info("StageMetrics...processing scene " + str(scene['scene_index']) + " test:" + test_name)
					test_folder = scene['tests'][test_name]['test_path']
					if not os.path.exists(test_folder):
						logging.warning("StageMetrics...WARNING test folder does not exist for test: " + test_folder)
						continue

					p = mp.Process(target=self.CollectMetricsForScene, args=(scene['scene_index'], setcopy, scene, test_name, hr_folder, test_folder))
					processes.append(p)
					p.start()

					if (len(processes) == settings.set.max_num_processes):
						# we reached the maximum number of processes.  Wait until 1 finishes
						q = processes.pop(0)
						q.join()

					if (sceneindex == lastsceneindex and test_name == lasttestname):
						# Last scene is in processes.  Finish all joins
						[p.join() for p in processes]
						logging.info("StageMetrics..metrics collection complete for batch")
						processes.clear()

			# copy back
			settings.set = copy.deepcopy(setcopy)

		else:
			for sceneindex in sceneindices:
				scene = database.db.getScene(sceneindex)

				if settings.set.metrics_test_name != "" and settings.set.metrics_test_name not in scene['tests']:
					logging.info("StageMetrics...skipping scene " + str(scene['scene_index']) + " no test found with name: " + settings.set.metrics_test_name)
					continue

				if scene['cropped']:
					hr_folder = scene['cropped_path']
				else:
					hr_folder = scene['hr_path']

				if not os.path.exists(hr_folder):
					logging.warning("StageMetrics...WARNING HR folder does not exist for scene: " + hr_folder)
					continue

				for test_name in scene['tests']:

					if settings.set.metrics_test_name != "" and test_name != settings.set.metrics_test_name:
						continue

					if 'test_path' not in scene['tests'][test_name]:
						logging.info("StageMetrics...skipping scene " + str(scene['scene_index'] ) + " because test path not found")
						continue

					logging.info("StageMetrics...processing scene " + str(scene['scene_index']) + " test:" + test_name)
					test_folder = scene['tests'][test_name]['test_path']
					if not os.path.exists(test_folder):
						logging.warning("StageMetrics...WARNING test folder does not exist for test: " + test_folder)
						continue

					self.CollectMetricsForScene(scene['scene_index'], settings.set, scene,  test_name, hr_folder, test_folder)

		# read metrics and update the database
		for sceneindex in sceneindices:
			scene = database.db.getScene(sceneindex)

			if settings.set.metrics_test_name != "" and settings.set.metrics_test_name not in scene['tests']:
				logging.info("StageMetrics...skipping scene " + str(scene['scene_index']) + " because no test found with name: " + settings.set.metrics_test_name)
				continue

			for test_name in scene['tests']:

				if settings.set.metrics_test_name != "" and test_name != settings.set.metrics_test_name:
					continue

				test_folder = scene['tests'][test_name]['test_path']

				metrics_path = os.path.join(test_folder, "metrics.json")

				filepath = Path(metrics_path)
				if not filepath.is_file():
					logging.warning("WARNING - metrics data file not found:" + metrics_path)
					continue

				# logging.info("Opening database: " + self.database_filename)
				with open(metrics_path, 'r') as f:
					metrics_data = json.load(f)

				# save the metric data in the scene
				if test_name not in scene['tests']:
					scene['tests'][test_name] = {}
				scene['tests'][test_name].update(metrics_data)

				logging.info("StageMetrics.. Summary for scene: " + test_folder + " test: " + test_name)
				if 'psnr_yuv' in scene['tests'][test_name]:
					logging.info("  PSNR="   + "{0:8.4f}".format(scene['tests'][test_name]['psnr_yuv']))
				if 'ssim_yuv' in scene['tests'][test_name]:
					logging.info("  SSIM=" + "{0:8.4f}".format(scene['tests'][test_name]['ssim_yuv']))
				if 'msssim' in scene['tests'][test_name]:
					logging.info("  MSSSIM=" + "{0:8.4f}".format(scene['tests'][test_name]['msssim']))
				if 'vmaf' in scene['tests'][test_name]:
					logging.info("  VMAF=" + "{0:8.4f}".format(scene['tests'][test_name]['vmaf']))
				if 'gmaf' in scene['tests'][test_name]:
					logging.info("  GMAF=" + "{0:8.4f}".format(scene['tests'][test_name]['gmaf']))
				if 'lpips' in scene['tests'][test_name]:
					logging.info("  LPIPS=" + "{0:8.4f}".format(scene['tests'][test_name]['lpips']))
				if 'haarpsi' in scene['tests'][test_name]:
					logging.info("  HAARPSI="+"{0:8.4f}".format(scene['tests'][test_name]['haarpsi']))

		database.db.save()
		logging.info("StageMetrics..complete")
