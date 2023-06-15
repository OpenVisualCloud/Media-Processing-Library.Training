# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import time
st=time.time()

import settings
import database
import coloredlogs, logging
import os
import sys
from tendo import singleton
from os import path
from jsonargparse import ArgumentParser, ActionConfigFile, ParserError
from stage_initial     import StageInitial
from stage_download    import StageDownload
from stage_scenedetect import StageSceneDetect
from stage_classify    import StageClassify
from stage_split       import StageSplit
from stage_crop        import StageCrop
from stage_downscale   import StageDownscale
from stage_augment     import StageAugment
from stage_train       import StageTrain
from stage_test        import StageTest
from stage_metrics     import StageMetrics
from stage_log         import StageLog
from stage_compare     import StageCompare
from stage_plot        import StagePlot
from stage_final       import StageFinal

class FAIME:

	def __init__(self,args=None):
		self.ConfigureParameters(args)

		self.stages = []
		self.stages.append(StageInitial())
		self.stages.append(StageDownload())
		self.stages.append(StageSceneDetect())
		self.stages.append(StageClassify())
		self.stages.append(StageSplit())
		self.stages.append(StageCrop())
		self.stages.append(StageDownscale())
		self.stages.append(StageAugment())
		self.stages.append(StageTrain())
		self.stages.append(StageTest())
		self.stages.append(StageMetrics())
		self.stages.append(StageLog())
		self.stages.append(StageCompare())
		self.stages.append(StagePlot())
		self.stages.append(StageFinal())
		pass

	def ConfigureParameters(self,args=None):
		self.parser = ArgumentParser(env_prefix='FAIME', default_env=True, description='Prepare and train Video SuperResolution ML models', default_config_files=['config.json'],parser_mode='jsonnet')
		self.parser.add_argument('--cfg',                                                                                              type=str,       help="configuration file\n\n", action=ActionConfigFile)
		self.parser.add_argument('--initial_stage',                        '-st_in',   metavar='', default='Initial',                  type=str,       help='Stage to start at: (Initial, Download, SceneDetect, Classify, Split, Augment, Downscale, Train, Test, Metrics, Log, Compare, Plot, Final)')
		self.parser.add_argument('--final_stage',                          '-st_fi',   metavar='', default='Initial',                  type=str,       help='Stage to end at: (Initial, Download, SceneDetect, Classify, Split, Augment, Downscale, Train, Test, Metrics, Log, Compare, Plot, Final)')
		self.parser.add_argument('--project_folder',                       '-prj_fo',  metavar='', default='',                         type=str,       help='Project folder. If blank, current folder will be used.')
		self.parser.add_argument('--models_folder',                        '-mo_fo',   metavar='', default='',                         type=str,       help='models folder. If blank, a models subfolder in the project folder will be used.')
		self.parser.add_argument('--videos_folder',                        '-vi_fo',   metavar='', default='',                         type=str,       help='videos folder. If blank, a videos subfolder in the project folder will be used.')
		self.parser.add_argument('--individual_scenes',                    '-ind_sc',  metavar='', default="",                         type=str,       help="Individual Scenes to process- separate values with commas (e.g. \"0, 4, 8\")")
		self.parser.add_argument('--multiprocess',                         '-mp',      metavar='', default=True,                       type=bool,      help="Use multiprocessing when executing stages (scenes will be processed in parallel")
		self.parser.add_argument('--max_num_processes',                    '-mp_max',  metavar='', default=40,                         type=int,       help='Maximum number of processes to spawn when doing multiprocessing (only currently used in Test and Metrics stages)')
		self.parser.add_argument('--system_report',                        '-sr',      metavar='', default=True,                       type=bool,      help='Ouptut a system report when script starts\n\n')

		self.parser.add_argument('--use_project_builder',                  '-pb_u',    metavar='', default=False,                      type=bool,      help='Use ProjectBuilder to create a project with the provided media folder')
		self.parser.add_argument('--project_builder_media_folder',         '-pb_mf',   metavar='', default='',                         type=str,       help='Path to media folder for ProjectBuilder')
		self.parser.add_argument('--project_builder_img_per_vid',          '-pb_iv',   metavar='', default=-1,                         type=int,       help='Number of frames for ProjectBuilder to extract from videos in media folder')
		self.parser.add_argument('--project_builder_percent_frames',       '-pb_pf',   metavar='', default=-1.0,                       type=float,     help='Percent of frames for ProjectBuilder to extract from videos in media folder')
		self.parser.add_argument('--project_builder_reset_project',        '-pb_rp',   metavar='', default=False,                      type=bool,       help='Reset project folder using project builder')

		self.parser.add_argument('--database_reset',                       '-db_re',   metavar='', default=False,                      type=bool,      help='Clear the database before running the pipeline. All scenes will be removed. All folders (except HR and videos) will be deleted.')
		self.parser.add_argument('--database_build',                       '-db_bu',   metavar='', default=False,                      type=bool,      help='build the database based on content in the project folder')
		self.parser.add_argument('--database_remove_scenes',               '-db_rms',  metavar='', default="",                         type=str,       help='a list of individual scenes to remove in the database - separate values with commas (e.g. "0, 1, 4").\n\n')

		self.parser.add_argument('--download_skip_stage',                  '-dl_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--download_video_urls',                  '-dl_url',  metavar='', default='',                         type=str,       help='Video URLs to download (comma or space delimited')
		self.parser.add_argument('--download_video_vimeos',                '-dl_vim',  metavar='', default='',                         type=str,       help='Video viemos to download (comma or space delimited')
		self.parser.add_argument('--download_video_youtubes',              '-dl_vyt',  metavar='', default='',                         type=str,       help='Video youtubes to download (comma or space delimited')
		self.parser.add_argument('--download_overwrite',                   '-dl_ov',   metavar='', default=False,                      type=bool,      help='Should the download operation overrwrite existing video files with same filename? (True, False)\n\n')

		self.parser.add_argument('--scenedetect_skip_stage',               '-sd_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--scenedetect_video_filters',            '-sd_vf',   metavar='', default="*.mp4,*.y4m",              type=str,       help='comma delimited list of filters used for selecting video files (e.g. "*.mp4,*.y4m" ')
		self.parser.add_argument('--scenedetect_max_video_length',         '-sd_mvl',  metavar='', default=-1,                         type=int,       help='Maximum video length to process (in frames) -1 = process entire video.  Used in histogram algorithm only')
		self.parser.add_argument('--scenedetect_algorithm',                '-sd_al',   metavar='', default="histogram",                type=str,       help='Which algorithm to use (histogram, content=PySceneDetect Content, threshold=PySceneDetect Threshold)')
		self.parser.add_argument('--scenedetect_threshold',                '-sd_th',   metavar='', default=40,                         type=int,       help='Threshold value to detect scene change.  Equal to the delta of two frames''s hsv average value (e.g. 30)')
		self.parser.add_argument('--scenedetect_downscale',                '-sd_ds',   metavar='', default=0,                          type=int,       help='Downscale factor.  The frame is downscaled by this factor before it is used for scene detection.  If equal to 0, an optimal downscalefactor is used')
		self.parser.add_argument('--scenedetect_min_accept_length',        '-sd_mal',  metavar='', default=120,                        type=int,       help='Minimum Scene Length (in frames) accepted after detection (e.g. 120)')
		self.parser.add_argument('--scenedetect_max_accept_length',        '-sd_mxl',  metavar='', default=200,                        type=int,       help='Maximum Scene Length (in frames) accepted after detection (e.g. 200) (-1 = no maximum)')
		self.parser.add_argument('--scenedetect_histogram_block_size',     '-sd_hbs',  metavar='', default=32,                         type=int,       help='Histogram block size (in pixels)')
		self.parser.add_argument('--scenedetect_histogram_num_bins',       '-sd_hnb',  metavar='', default=256,                        type=int,       help='Number of bins to calculate in histogram')
		self.parser.add_argument('--scenedetect_histogram_neighbor_size',  '-sd_hns',  metavar='', default=7,                          type=int,       help='Number of frames on each side of the frame to use as a neighborhood (e.g. 3 -> f-3, f-2, f-1, f, f+1, f+2, f+3)')
		self.parser.add_argument('--scenedetect_histogram_threshold',      '-sd_hth',  metavar='', default=1.0,                        type=float,     help='Threshold for scene detection against previous frame')
		self.parser.add_argument('--scenedetect_histogram_nei_threshold',  '-sd_hnth', metavar='', default=0.8,                        type=float,     help='Threshold for scene detection against neighborhood of frames')
		self.parser.add_argument('--scenedetect_overwrite',                '-sd_ov',   metavar='', default=False,                      type=bool,      help='Should the scenedetect operation overrwrite existing scenes detected with same filename? (True, False)')
		self.parser.add_argument('--scenedetect_max_scenes_per_video',     '-sd_mxsv', metavar='', default=-1,                         type=int,       help='Maximum number of scenes to be detected from a single video (-1 = unlimited)\n\n')
		self.parser.add_argument('--scenedetect_max_scenes',               '-sd_mx',   metavar='', default=-1,                         type=int,       help='Maximum number of scenes to be detected from all videos (-1 = unlimited)\n\n')

		self.parser.add_argument('--classify_skip_stage',                  '-cl_sk',   metavar='', default=True,                       type=bool,      help='Skip this stage\n\n')

		self.parser.add_argument('--split_skip_stage',                     '-sp_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--split_overwrite',                      '-sp_ov',   metavar='', default=False,                      type=bool,      help='Should the split stage overrwrite existing scene folders? (True, False)')
		self.parser.add_argument('--split_downscalefactor',                '-sp_sf',   metavar='', default=1,                          type=int,       help='Used to downscale the split frames when creating the high resolution - should be an integer values (eg: 2,3,or 4)')
		self.parser.add_argument('--split_downscale_algorithm',            '-sp_al',   metavar='', default=0,                          type=int,       help='OpenCV Algorithm used to downscale: 0=bilinear, 1=nearest, 2=area, 3=bicubic, 4=Lanczos\n\n')

		self.parser.add_argument('--augment_skip_stage',                   '-au_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--augment_seed',                         '-au_sd',   metavar='', default=0,                          type=int,       help='Seed for augmentation effects (e.g. 0)')
		self.parser.add_argument('--augment_random_effects',               '-au-re',   metavar='', default="",                         type=str,       help='Comma delimited list of random effects to choose from (e.g. gaussian_low_std, burst_low_per)')
		self.parser.add_argument('--augment_effect',                       '-au_ef',   metavar='', default="",                         type=str,       help='Augmentation effect: '
																																							'salt_and_pepper, '
																																							'gaussian_low_std, '
																																							'gaussian_medium_std, '
																																							'gaussian_high_std, '
																																							'block_2, '
																																							'block_4, '
																																							'block_8, '
																																							'block_16, '
																																							'block_32, '
																																							'block_64, '
																																							'burst_low_per, '
																																							'burst_medium_per, '
																																							'burst_high_per, '
																																							'poisson_low, '
																																							'poisson_medium, '
																																							'poisson_high, '
																																							'laplace_low, '
																																							'laplace_medium, '
																																							'laplace_high, '
																																							'jpegcompression_low, '
																																							'jpegcompression_medium, '
																																							'jpegcompression_high, '
																																							'motionblur_low, '
																																							'motionblur_medium, '
																																							'motionblur_high, '
																																							'sharpen_low, '
																																							'sharpen_medium, '
																																							'sharpen_high, '
																																							'random\n\n')


		self.parser.add_argument('--crop_skip_stage',                      '-cr_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--crop_overwrite',                       '-cr_ov',   metavar='', default=True,                       type=bool,      help='Should the crop operation overrwrite existing files with same filename? (True, False)')
		self.parser.add_argument('--crop_num_frames',                      '-cr_nf',   metavar='', default=-1,                         type=int,       help='Number of middle frames to crop the scene to. -1= use all frames')
		self.parser.add_argument('--crop_fraction',                        '-cr_fr',   metavar='', default=1.0,                        type=float,     help='Center percentage of each frame to crop.  1.0 = use entire frame')
		self.parser.add_argument('--crop_divisible',                       '-cr_div',  metavar='', default=1,                          type=int,       help='Crop frame size to be divisible by a value (e.g. =2 width, height are divisible by 2)/n/n')

		self.parser.add_argument('--downscale_skip_stage',                 '-ds_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--downscale_overwrite',                  '-ds_ov',   metavar='', default=True,                       type=bool,      help='Should the downscale stage overrwrite existing LR folders? (True, False)')
		self.parser.add_argument('--downscale_scalefactor',                '-ds_sf',   metavar='', default=4,                          type=int,       help='Used to downscale the high resolution original frames - should be an integer values (eg: 2,3,or 4)')
		self.parser.add_argument('--downscale_algorithm',                  '-ds_al',   metavar='', default=3,                          type=int,       help='OpenCV Algorithm used to downscale: 0=bilinear, 1=nearest, 2=area, 3=bicubic, 4=Lanczos, 5=Blur, 6=Random (will use random choice of 0-5)')
		self.parser.add_argument('--downscale_format',                     '-ds_fmt',  metavar='', default="PNG",                      type=str,       help='File format for downscaled image files: PNG or JPEG')
		self.parser.add_argument('--downscale_JPEG_quality',               '-ds_jq',   metavar='', default=95,                         type=int,       help='JPEG quality for downscaled JPEG image files (0-100).  Use -1 for random quality.')
		self.parser.add_argument('--downscale_random_per_scene',           '-ds_rps',  metavar='', default=False,                      type=bool,      help='Apply random settings (downscale algorithm and jpeg quality) per scene rather than per image.')
		self.parser.add_argument('--downscale_create_mkv',                 '-ds_mkv',  metavar='', default=False,                      type=bool,      help='Create an MKV (Matroska) video from the downscaled frames. This file will contain uncompressed frames.\n\n')

		self.parser.add_argument('--train_skip_stage',                     '-tr_sk',   metavar='', default=True,                       type=bool,      help='Skip this stage')
		self.parser.add_argument('--train_algorithm',                      '-tr_al',   metavar='', default="RAISR",                    type=str,       help='which SR algorithm')
		self.parser.add_argument('--train_raisr_first_pass',               '-tr_fp',   metavar='', default=True,                       type=bool,      help='Set to true to train first pass filters with RAISR')
		self.parser.add_argument('--train_raisr_second_pass',              '-tr_sp',   metavar='', default=False,                      type=bool,      help='Set to true to train second pass filters with RAISR')
		self.parser.add_argument('--train_raisr_sharpen',                  '-tr_s',     metavar='', default=1,                          type=float,     help='Sharpen value for 2 pass training')
		self.parser.add_argument('--train_raisr_scale',                    '-tr_rs',   metavar='', default=2,                          type=int,       help='Raisr upscale factor')
		self.parser.add_argument('--train_raisr_patch_size',               '-tr_rps',  metavar='', default=11,                         type=int,       help='Raisr patch size')
		self.parser.add_argument('--train_raisr_gradient_size',            '-tr_rgs',  metavar='', default=9,                          type=int,       help='Raisr gradient size')
		self.parser.add_argument('--train_raisr_angle_quantization',       '-tr_raq',  metavar='', default=24,                         type=int,       help='Raisr angle quantization')
		self.parser.add_argument('--train_raisr_strength_quantization',    '-tr_rsq',  metavar='', default=3,                          type=int,       help='Raisr strength quantization')
		self.parser.add_argument('--train_raisr_coherence_quantization',   '-tr_rcq',  metavar='', default=3,                          type=int,       help='Raisr coherence quantization')
		self.parser.add_argument('--train_raisr_filterpath',   			   '-tr_rfp',  metavar='', default="",                         type=str,       help='Raisr trained filter path')
		self.parser.add_argument('--train_raisr_input_folder',             '-tr_rif',  metavar='', default="",                         type=str,       help="Raisr input folder (if not set, use all HR images")
		self.parser.add_argument('--train_raisr_bit_depth',                '-tr_bit',  metavar='', default='8',                        type=str,       help='Raisr training bit depth for HRImages')
		self.parser.add_argument('--test_skip_stage',                      '-te_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--test_cpu_only',                        '-te_cpu',  metavar='', default=False,                      type=bool,      help="True: run testing on CPU only. False: run testing on GPU only if CUDA is detected.")
		self.parser.add_argument('--test_clear_tests',                     '-te_clr',  metavar='', default=False,                      type=bool,      help='Remove all test results for all the scenes, prior to running any new tests')
		self.parser.add_argument('--test_algorithm',                       '-te_alg',  metavar='', default='',                         type=str,       help="which SR algorithm: RAISR_ffmpeg, nearest_neighbor,bilinear, bicubic, area, or lanczos. If blank, no test will be run.")
		self.parser.add_argument('--test_difference_style',                '-te_diff', metavar='', default="None",                     type=str,       help='Style of difference file NONE=none RGB=Color JET=Blue-Red RAINBOW=Red-Green_purple HOT=Black-Red-Yellow  For more refer to: https://docs.opencv.org/4.5.2/d3/d50/group__imgproc__colormap.html')
		self.parser.add_argument('--test_difference_multiplier',           '-te_difm', metavar='', default=1.0,                        type=float,     help='Difference multiplier - scales differences to make them more visible in difference image')
		self.parser.add_argument('--test_overwrite',                       '-te_ov',   metavar='', default=False,                      type=bool,      help='Should the test stage overrwrite existing folders? (True, False)')
		self.parser.add_argument('--test_raisr_input_folder',              '-te_rif',  metavar='', default="",                         type=str,       help="Raisr testing input folder (if not set, use default Jaladi filters")
		self.parser.add_argument('--test_user_trained_raisr',              '-te_utr',  metavar='', default=False,                      type=bool,      help="Use output of train stage for test stage")
		self.parser.add_argument('--test_RaisrFF_filter_path',             '-te_rfff', metavar='', default='',                         type=str,       help='Location of Raisr Filters')
		self.parser.add_argument('--test_RaisrFF_scale_interpolation',     '-te_rfsc', metavar='', default="lanczos",                  type=str,       help='Scale interpoluation used in Raisr (lanczos or bicubic)')
		self.parser.add_argument('--test_RaisrFF_bits',                    '-te_rfbi', metavar='', default=8,                          type=int,       help='Raisr FFmpeg Plugin bits')
		self.parser.add_argument('--test_RaisrFF_range',                   '-te_rfr',  metavar='', default='video',                    type=str,       help='Raisr FFmpeg Plugin range')
		self.parser.add_argument('--test_RaisrFF_threadcount',             '-te_rftc', metavar='', default=120,                        type=int,      help='Raisr FFmpeg Plugin threadcount')
		self.parser.add_argument('--test_RaisrFF_blending',                '-te_rfbl', metavar='', default=1,                          type=int,       help='Raisr FFmpeg Plugin blending')
		self.parser.add_argument('--test_RaisrFF_passes',                  '-te_rfp',  metavar='', default=1,                          type=int,       help='Raisr FFmpeg Plugin passes')
		self.parser.add_argument('--test_RaisrFF_mode',                    '-te_rfm',  metavar='', default=1,                          type=str,       help='Raisr FFmpeg Plugin mode')
		self.parser.add_argument('--test_upscale_factor',                  '-te_usf',  metavar='', default=4,                          type=int,       help='Scale factor to use when testing-inferring  (e.g. 0, 2 or 4),Currently RAISR hardcoded to 2x.  If 0, calculate scalefactor from HR and LR frame size.')
		self.parser.add_argument('--test_name',                            '-te_nm',   metavar='', default="",                         type=str,       help='Name of the test run.  If blank, the test algorithm will be used for the name.  Used as the folder name for the SR test results.')
		self.parser.add_argument('--test_tile',                            '-te_ti',   metavar='', default=False,                      type=bool,      help="Break the LR frames into a series of tiles, run the test on the tiles, and then stitch back the results.")
		self.parser.add_argument('--test_tile_division',                   '-te_tid',  metavar='', default=4,                          type=int,       help="How to divide the LR frames for tiling (applied to both width and height, e.g. 4=16 tiles).")
		self.parser.add_argument('--test_tile_overlap',                    '-te_tio',  metavar='', default=5,                          type=int,       help="How many pixels to overlap the tiles")
		self.parser.add_argument('--test_video_create',                    '-te-vidc', metavar='', default=False,                      type=bool,      help="Create video of test frames (mp4 format)")
		self.parser.add_argument('--test_video_quality',                   '-te-vidq', metavar='', default=5,                          type=int,       help="Quality of created video (0-9).\n\n")

		self.parser.add_argument('--metrics_skip_stage',                   '-me_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--metrics_overwrite',                    '-me_ov',   metavar='', default=False,                      type=bool,      help='Should the metrics stage overwrite existing results? (True, False)')
		self.parser.add_argument('--metrics_psnr',                         '-me_psnr', metavar='', default=False,                      type=bool,      help='Calculate PSNR metrics')
		self.parser.add_argument('--metrics_ssim',                         '-me_ssim', metavar='', default=False,                      type=bool,      help='Calculate SSIM metrics')
		self.parser.add_argument('--metrics_msssim',                       '-me_mss',  metavar='', default=False,                      type=bool,      help='Calculate MSSSIM metrics')
		self.parser.add_argument('--metrics_vmaf',                         '-me_vmaf', metavar='', default=False,                      type=bool,      help='Calculate VMAF metrics')
		self.parser.add_argument('--metrics_vmaf_preload',                 '-me_vmafp',metavar='', default=True,                       type=bool,      help='Preload VMAF library on linux when calculating VMAF metrics')
		self.parser.add_argument('--metrics_vmaf_delete_yuvs',             '-me_vmafd',metavar='', default=True,                       type=bool,      help='Delete the composite scene yuv files after calculating vmaf metric')
		self.parser.add_argument('--metrics_single_frame',                 '-me_vmafs',metavar='', default=False,                      type=bool,      help='Calculate VMAF for each frame independently, rather than as a sequence')
		self.parser.add_argument('--metrics_gmaf',                         '-me_gmaf', metavar='', default=False,                      type=bool,      help='Calculate VMAF metrics')
		self.parser.add_argument('--metrics_gmaf_delete_yuvs',             '-me_gmafd',metavar='', default=True,                       type=bool,      help='Delete the composite scene yuv files after calculating gmaf metric')
		self.parser.add_argument('--metrics_lpips',                        '-me_lpips',metavar='', default=False,                      type=bool,      help='Calculate LPIPS metrics')
		self.parser.add_argument('--metrics_haarpsi',                      '-me_psi',  metavar='', default=False,                      type=bool,      help='Calculate Haar Wavelet-Based Perceptual Similarity Index')
		self.parser.add_argument('--metrics_test_name',                    '-me_nm',   metavar='', default="",                         type=str,       help='Name of the test to collect metrics on.  If blank, metrics for all tests run will be calculated.')
		self.parser.add_argument('--metrics_border_crop',                  '-me_bc',   metavar='', default=False,                      type=bool,      help='Perform a border crop before calculating metrics.  Cropped area is a 32x32 multiple and centered in the original frame.\n\n')

		self.parser.add_argument('--log_skip_stage',                       '-log_sk',  metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--log_logfilename',                      '-log_fn',  metavar='', default="log.csv",                  type=str,       help='Name for the log file (saved to csv format)')
		self.parser.add_argument('--log_overwrite',                        '-log_ov',  metavar='', default=False,                      type=bool,      help='Overwrite an existing log file if found, otherwise append')
		self.parser.add_argument('--log_test_name',                        '-log_nm',  metavar='', default="",                         type=str,       help='Name of the test to log.  If blank, all tests will be logged\n\n')

		self.parser.add_argument('--compare_skip_stage',                   '-cp_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--compare_overwrite',                    '-cp_ov',   metavar='', default=False,                      type=bool,      help='Should the compare stage overwrite existing results? (True, False)')
		self.parser.add_argument('--compare_visualqualitytool',            '-cp_vqt',  metavar='', default="none",                     type=str,       help='Visual Quality Tool configuration creation: absolute (use absolute paths), relative (use relative paths), none (do not create)')
		self.parser.add_argument('--compare_original',                     '-cp_org',  metavar='', default=True,                       type=bool,      help='Include original (HR) in comparision')
		self.parser.add_argument('--compare_nearest_neighbor',             '-cp_nn',   metavar='', default=False,                      type=bool,      help='Include nearest neighbor scaling in comparision')
		self.parser.add_argument('--compare_bilinear',                     '-cp_bi',   metavar='', default=False,                      type=bool,      help='Include bilinear scaling in comparision')
		self.parser.add_argument('--compare_area',                         '-cp_ar',   metavar='', default=False,                      type=bool,      help='Include area scaling in comparision')
		self.parser.add_argument('--compare_bicubic',                      '-cp_bc',   metavar='', default=False,                      type=bool,      help='Include bicubic scaling in comparision')
		self.parser.add_argument('--compare_lanczos',                      '-cp_lc',   metavar='', default=False,                      type=bool,      help='Include Lanczos scaling in comparision')
		self.parser.add_argument('--compare_tests',                        '-cp_te',   metavar='', default=True,                       type=bool,      help='Include tests in comparision')
		self.parser.add_argument('--compare_descriptions',                 '-cp_de',   metavar='', default=True,                       type=bool,      help='Include descriptions in comparision sheet')
		self.parser.add_argument('--compare_numcolumns',                   '-cp_ncol', metavar='', default=-1,                         type=int,       help='Number of columns for the grids used in the comparision sheet and page.  -1 = single row')
		self.parser.add_argument('--compare_roifilename',                  '-cp_roifn',metavar='', default="compare_regions.txt",      type=str,       help='ROR filename - csv formatted file.  Each line contains scenename,frame,X,Y,width,height (e.g. "calendar, 4, 20, 80, 200, 200\").  Leave blank for no ROI generation.  Use misc/compare_regions.txt as example file.')

		self.parser.add_argument('--plot_skip_stage',                      '-pl_sk',   metavar='', default=False,                      type=bool,      help='Skip this stage')
		self.parser.add_argument('--plot_test_name',                       '-pl_nm',   metavar='', default="",                         type=str,       help='Name of the test to plot.  If blank, all tests will be plotted')
		self.parser.add_argument('--plot_metrics',                         '-pl_me',   metavar='', default="psnr_yuv,ssim,msssim,vmaf",type=str,       help='Which metrics to plot.  Choose from:  psnr_yuv, psnr_y, psnr_u, psnr_v, ssim, msssim, vmaf, vmaf_adm2, vmaf_adm_scale0, vmaf_adm_scale1,'
																																					   'vmaf_adm_scale2, vmaf_adm_scale3, vmaf_adm_motion2, vmaf_adm_motion, vmaf_vif_scale0, vmaf_vif_scale1, vmaf_vif_scale2,'																																				   'vmaf_vif_scale3, average test time \n\n')
		self.arg = self.parser.parse_args(args=args)
		settings.set.InitializeFromParser(self.parser.parse_args(args=args))

	def Echo_Purpose(self):
		logging.info("Framework for AI Media Engineering - v1.0")
		logging.info("This framework will assist in preparing training data for Video SuperResolution algorithms")

	def Echo_System(self):
		import psutil
		import platform
		import utils
		from datetime import datetime
		import GPUtil
		from tabulate import tabulate
		import torch

		logging.info("=" * 40 + "System Information" + "=" * 40)
		uname = platform.uname()
		logging.info(f"System: {uname.system}")
		logging.info(f"Node Name: {uname.node}")
		logging.info(f"Release: {uname.release}")
		logging.info(f"Version: {uname.version}")
		logging.info(f"Machine: {uname.machine}")
		logging.info(f"Processor: {uname.processor}")

		# Boot Time
		logging.info("=" * 40 + "Boot Time" + "=" * 40)
		boot_time_timestamp = psutil.boot_time()
		bt = datetime.fromtimestamp(boot_time_timestamp)
		logging.info(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

		# let's print CPU information
		logging.info("=" * 40 + "CPU Info" + "=" * 40)
		# number of cores
		logging.info("Physical cores:" + str(psutil.cpu_count(logical=False)))
		logging.info("Total cores:" + str(psutil.cpu_count(logical=True)))
		# CPU frequencies
		cpufreq = psutil.cpu_freq()
		logging.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
		logging.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
		logging.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")
		# CPU usage
		logging.info("CPU Usage Per Core:")
		for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
			logging.info(f"Core {i}: {percentage}%")
		logging.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

		# GPU information
		logging.info("=" * 40 + "Nvidia GPU Details" + "=" * 40)
		try:
			gpus = GPUtil.getGPUs()
		except ValueError as e:
			logging.warning('NVIDIA SMI installed but no GPU/driver present')
			gpus = []
		list_gpus = []
		for gpu in gpus:
			# get the GPU id
			gpu_id = gpu.id
			# name of GPU
			gpu_name = gpu.name
			# get % percentage of GPU usage of that GPU
			gpu_load = f"{gpu.load * 100}%"
			# get free memory in MB format
			gpu_free_memory = f"{gpu.memoryFree}MB"
			# get used memory
			gpu_used_memory = f"{gpu.memoryUsed}MB"
			# get total memory
			gpu_total_memory = f"{gpu.memoryTotal}MB"
			# get GPU temperature in Celsius
			gpu_temperature = f"{gpu.temperature} °C"
			gpu_uuid = gpu.uuid
			list_gpus.append((
				gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
				gpu_total_memory, gpu_temperature, gpu_uuid
			))
		print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature", "uuid")))

		use_cuda = torch.cuda.is_available()
		if use_cuda:
			logging.info('CUDNN VERSION:' + str(torch.backends.cudnn.version()))
			logging.info('Number CUDA Devices:' + str(torch.cuda.device_count()))
			logging.info('CUDA Device Name:' + torch.cuda.get_device_name(0))
			logging.info('CUDA Device Total Memory [GB]:' + str(torch.cuda.get_device_properties(0).total_memory / 1e9))
		else:
			logging.info('CUDA not available')

		# Memory Information
		logging.info("=" * 40 + "Memory Information" + "=" * 40)
		# get the memory details
		svmem = psutil.virtual_memory()
		logging.info(f"Total: {utils.get_size(svmem.total)}")
		logging.info(f"Available: {utils.get_size(svmem.available)}")
		logging.info(f"Used: {utils.get_size(svmem.used)}")
		logging.info(f"Percentage: {svmem.percent}%")
		logging.info("=" * 20 + "SWAP" + "=" * 20)
		# get the swap memory details (if exists)
		swap = psutil.swap_memory()
		logging.info(f"Total: {utils.get_size(swap.total)}")
		logging.info(f"Free: {utils.get_size(swap.free)}")
		logging.info(f"Used: {utils.get_size(swap.used)}")
		logging.info(f"Percentage: {swap.percent}%")

		# Disk Information
		logging.info("=" * 40 + "Disk Information" + "=" * 40)
		logging.info("Partitions and Usage:")
		# get all disk partitions
		partitions = psutil.disk_partitions()
		for partition in partitions:
			logging.info(f"=== Device: {partition.device} ===")
			logging.info(f"  Mountpoint: {partition.mountpoint}")
			logging.info(f"  File system type: {partition.fstype}")
			try:
				partition_usage = psutil.disk_usage(partition.mountpoint)
			except PermissionError:
				# this can be catched due to the disk that
				# isn't ready
				continue
			logging.info(f"  Total Size: {utils.get_size(partition_usage.total)}")
			logging.info(f"  Used: {utils.get_size(partition_usage.used)}")
			logging.info(f"  Free: {utils.get_size(partition_usage.free)}")
			logging.info(f"  Percentage: {partition_usage.percent}%")
		# get IO statistics since boot
		disk_io = psutil.disk_io_counters()
		logging.info(f"Total read: {utils.get_size(disk_io.read_bytes)}")
		logging.info(f"Total write: {utils.get_size(disk_io.write_bytes)}")

		# Network information
		logging.info("=" * 40 + "Network Information" + "=" * 40)
		# get all network interfaces (virtual and physical)
		if_addrs = psutil.net_if_addrs()
		for interface_name, interface_addresses in if_addrs.items():
			for address in interface_addresses:
				logging.info(f"=== Interface: {interface_name} ===")
				if str(address.family) == 'AddressFamily.AF_INET':
					logging.info(f"  IP Address: {address.address}")
					logging.info(f"  Netmask: {address.netmask}")
					logging.info(f"  Broadcast IP: {address.broadcast}")
				elif str(address.family) == 'AddressFamily.AF_PACKET':
					logging.info(f"  MAC Address: {address.address}")
					logging.info(f"  Netmask: {address.netmask}")
					logging.info(f"  Broadcast MAC: {address.broadcast}")
		# get IO statistics since boot
		net_io = psutil.net_io_counters()
		logging.info(f"Total Bytes Sent: {utils.get_size(net_io.bytes_sent)}")
		logging.info(f"Total Bytes Received: {utils.get_size(net_io.bytes_recv)}")

	def Run(self):
		self.Echo_Purpose()

		if settings.set.system_report:
			self.Echo_System()

		# make sure project folder exists
		if settings.set.project_folder != "":
			if not path.exists(settings.set.project_folder):
				logging.warning("WARNING =..project folder does not exist. A new project folder will be created: "+settings.set.project_folder)
				os.mkdir(settings.set.project_folder)
			settings.set.absolute_project_folder = path.abspath(settings.set.project_folder)
		else:
			settings.set.absolute_project_folder = path.abspath(path.curdir)

		if settings.set.use_project_builder:
			import project_builder.project_builder as pb
			project_builder_parameters = []

			# Add image per video parameter
			if settings.set.project_builder_img_per_vid > -1:
				project_builder_parameters.extend(['--img_per_vid', str(settings.set.project_builder_img_per_vid)])

			# Add percent frames parameter
			elif settings.set.project_builder_percent_frames > 0.0 and settings.set.project_builder_percent_frames <= 1.00:
				project_builder_parameters.extend(['--percent_frames', str(settings.set.project_builder_percent_frames)])

			# Add media and project folder parameters
			project_builder_parameters.extend(['--media_folder', settings.set.project_builder_media_folder,
												'--project_folder', settings.set.project_folder])

			# Add reset project parameter
			project_builder_parameters.extend(['--reset_project', str(settings.set.project_builder_reset_project)])

			# Call project builder with specified parameters
			pb.main(args=project_builder_parameters)

		settings.set.absolute_modules_folder = os.path.dirname(os.path.abspath(__file__))
		settings.set.absolute_resources_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'resources')
		settings.set.absolute_HR_folder = path.join(settings.set.absolute_project_folder, "HR")
		settings.set.absolute_LR_folder = path.join(settings.set.absolute_project_folder, "LR")
		settings.set.absolute_CHR_folder = path.join(settings.set.absolute_project_folder, "CHR")
		settings.set.absolute_test_folder = path.join(settings.set.absolute_project_folder, "test")
		settings.set.absolute_train_folder = path.join(settings.set.absolute_project_folder, "train")
		settings.set.absolute_contact_sheets_folder = path.join(settings.set.absolute_project_folder, "contact_sheets")
		settings.set.absolute_compare_folder = path.join(settings.set.absolute_project_folder, "compare")
		settings.set.absolute_vqt_folder = path.join(settings.set.absolute_project_folder, "vqt")

		if settings.set.models_folder != "":
			if not path.exists(settings.set.models_folder):
				logging.warning("WARNING =..models folder does not exist.  " + settings.set.models_folder)
				return
			settings.set.absolute_models_folder = settings.set.models_folder
		else:
			settings.set.absolute_models_folder = path.join(settings.set.absolute_project_folder, "models")

		if settings.set.videos_folder != "":
			if not path.exists(settings.set.videos_folder):
				logging.warning("WARNING =..videos folder does not exist.  " + settings.set.videos_folder)
				return
			settings.set.absolute_videos_folder = settings.set.videos_folder
		else:
			settings.set.absolute_videos_folder = path.join(settings.set.absolute_project_folder, "videos")

		settings.set.absolute_plots_folder = path.join(settings.set.absolute_project_folder, "plots")

		if settings.set.database_reset:
			database.db.clear()

		if settings.set.database_build:
			database.db.build()
			database.db.make_scene_sheet()
		else:
			database.db.open()

		# list the contents of the database
		database.db.list()

		if len(settings.set.database_remove_scenes):
			database.db.removescenes(settings.set.database_remove_scenes)
			database.db.make_scene_sheet()

		# Clear the runtime arguments (except for the running script path) now that they've been processed. 
		sys.argv = [sys.argv[0]]

		# Start with the initial stage
		stage = settings.set.initial_stage
		while (stage <= settings.set.final_stage):
			self.stages[stage].ExecuteStage()
			stage = stage + 1
		logging.info("FAIME run complete")

def main(args=sys.argv[1:]):
	if 'unittest' not in sys.modules.keys():
		me = singleton.SingleInstance()

	logger = logging.getLogger(__name__)
	coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

	logging.info("Startup time: %.2fsec" % (time.time() - st))

	try:
		app = FAIME(args)
	except ParserError as e:
		logging.error('ERROR: {}'.format(e.args[0]))
		return
	app.Run()

if __name__ == "__main__":

	# make sure just a single instance of the script is running
	# https: // stackoverflow.com / questions / 380870 / make - sure - only - a - single - instance - of - a - program - is -running
	me = singleton.SingleInstance()

	# Configure Logging
	logger = logging.getLogger(__name__)
	coloredlogs.install(fmt='%(asctime)s - %(message)s', level='INFO')

	# Log startup time
	logging.info("Startup time: %.2fsec" % (time.time() - st))

	#logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
	try:
		app = FAIME()
	except ParserError as e:
		logging.error('ERROR: {}'.format(e.args[0]))
		exit()
	app.Run()