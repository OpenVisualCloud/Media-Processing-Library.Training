# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

class Settings:

	def __init__(self):
		self.download_filelist = ''
		pass

	def InitializeFromParser(self, parserargs):
		self.initial_stage                         = ['initial', 'download', 'scenedetect', 'classify', 'split', 'crop', 'downscale', 'augment', 'train', 'test', 'metrics', 'log', 'compare', 'plot', 'final'].index(parserargs.initial_stage.lower())
		self.final_stage                           = ['initial', 'download', 'scenedetect', 'classify', 'split', 'crop', 'downscale', 'augment', 'train', 'test', 'metrics', 'log', 'compare', 'plot', 'final'].index(parserargs.final_stage.lower())
		self.project_folder                        = parserargs.project_folder
		self.models_folder                         = parserargs.models_folder
		self.videos_folder                         = parserargs.videos_folder

		self.individual_scenes = []
		if len(parserargs.individual_scenes) > 0:
			self.individual_scenes = [int(item) for item in parserargs.individual_scenes.split(',')]
		self.multiprocess                          = parserargs.multiprocess
		self.max_num_processes                     = parserargs.max_num_processes

		self.database_reset                        = parserargs.database_reset
		self.database_build                        = parserargs.database_build
		self.database_remove_scenes = []
		if len(parserargs.database_remove_scenes) > 0:
			self.database_remove_scenes = [int(item) for item in parserargs.database_remove_scenes.split(',')]
		self.system_report                         = parserargs.system_report


		self.use_project_builder 				   = parserargs.use_project_builder
		self.project_builder_media_folder          = parserargs.project_builder_media_folder
		self.project_builder_img_per_vid           = parserargs.project_builder_img_per_vid
		self.project_builder_percent_frames        = parserargs.project_builder_percent_frames
		self.project_builder_reset_project         = parserargs.project_builder_reset_project

		self.download_skip_stage                   = parserargs.download_skip_stage
		self.download_video_urls                   = parserargs.download_video_urls
		self.download_video_vimeos                 = parserargs.download_video_vimeos
		self.download_video_youtubes               = parserargs.download_video_youtubes
		self.download_overwrite                    = parserargs.download_overwrite

		self.scenedetect_skip_stage                = parserargs.scenedetect_skip_stage
		self.scenedetect_video_filters             = []
		if len(parserargs.scenedetect_video_filters) > 0:
			self.scenedetect_video_filters         = [item for item in parserargs.scenedetect_video_filters.split(',')]
		self.scenedetect_max_video_length          = parserargs.scenedetect_max_video_length
		self.scenedetect_algorithm                 = parserargs.scenedetect_algorithm
		self.scenedetect_threshold                 = parserargs.scenedetect_threshold
		self.scenedetect_downscale                 = parserargs.scenedetect_downscale
		self.scenedetect_min_accept_length         = parserargs.scenedetect_min_accept_length
		self.scenedetect_max_accept_length         = parserargs.scenedetect_max_accept_length
		self.scenedetect_histogram_block_size      = parserargs.scenedetect_histogram_block_size
		self.scenedetect_histogram_num_bins        = parserargs.scenedetect_histogram_num_bins
		self.scenedetect_histogram_neighbor_size   = parserargs.scenedetect_histogram_neighbor_size
		self.scenedetect_histogram_threshold       = parserargs.scenedetect_histogram_threshold
		self.scenedetect_histogram_nei_threshold   = parserargs.scenedetect_histogram_nei_threshold
		self.scenedetect_overwrite                 = parserargs.scenedetect_overwrite
		self.scenedetect_max_scenes_per_video      = parserargs.scenedetect_max_scenes_per_video
		self.scenedetect_max_scenes                = parserargs.scenedetect_max_scenes

		self.classify_skip_stage                   = parserargs.classify_skip_stage

		self.split_skip_stage                      = parserargs.split_skip_stage
		self.split_overwrite                       = parserargs.split_overwrite
		self.split_downscalefactor                 = parserargs.split_downscalefactor
		self.split_downscale_algorithm             = parserargs.split_downscale_algorithm

		self.augment_skip_stage                    = parserargs.augment_skip_stage
		self.augment_seed                          = parserargs.augment_seed
		self.augment_effect                        = parserargs.augment_effect
		self.augment_random_effects = []
		if len(parserargs.augment_random_effects) > 0:
			self.augment_random_effects = [item for item in parserargs.augment_random_effects.split(',')]

		self.crop_skip_stage                       = parserargs.crop_skip_stage
		self.crop_overwrite                        = parserargs.crop_overwrite
		self.crop_num_frames                       = parserargs.crop_num_frames
		self.crop_fraction                         = parserargs.crop_fraction
		self.crop_divisible                        = parserargs.crop_divisible

		self.downscale_skip_stage                  = parserargs.downscale_skip_stage
		self.downscale_overwrite                   = parserargs.downscale_overwrite
		self.downscale_scalefactor                 = parserargs.downscale_scalefactor
		self.downscale_algorithm                   = parserargs.downscale_algorithm
		self.downscale_format                      = parserargs.downscale_format
		self.downscale_JPEG_quality                = parserargs.downscale_JPEG_quality
		self.downscale_random_per_scene            = parserargs.downscale_random_per_scene
		self.downscale_create_mkv                  = parserargs.downscale_create_mkv

		self.train_skip_stage                      = parserargs.train_skip_stage
		self.train_algorithm                       = parserargs.train_algorithm
		self.train_raisr_first_pass                = parserargs.train_raisr_first_pass
		self.train_raisr_second_pass               = parserargs.train_raisr_second_pass
		self.train_raisr_sharpen                   = parserargs.train_raisr_sharpen
		self.train_raisr_scale                     = parserargs.train_raisr_scale
		self.train_raisr_patch_size                = parserargs.train_raisr_patch_size
		self.train_raisr_gradient_size             = parserargs.train_raisr_gradient_size
		self.train_raisr_angle_quantization        = parserargs.train_raisr_angle_quantization
		self.train_raisr_strength_quantization     = parserargs.train_raisr_strength_quantization
		self.train_raisr_coherence_quantization    = parserargs.train_raisr_coherence_quantization
		self.train_raisr_filterpath    			   = parserargs.train_raisr_filterpath
		self.train_raisr_input_folder              = parserargs.train_raisr_input_folder
		self.train_raisr_bit_depth                 = parserargs.train_raisr_bit_depth



		self.test_skip_stage                       = parserargs.test_skip_stage
		self.test_cpu_only                         = parserargs.test_cpu_only
		self.test_clear_tests                      = parserargs.test_clear_tests
		self.test_algorithm                        = parserargs.test_algorithm
		self.test_difference_style                 = parserargs.test_difference_style
		self.test_difference_multiplier            = parserargs.test_difference_multiplier
		self.test_overwrite                        = parserargs.test_overwrite
		self.test_raisr_input_folder               = parserargs.test_raisr_input_folder
		self.test_user_trained_raisr               = parserargs.test_user_trained_raisr
		self.test_RaisrFF_filter_path              = parserargs.test_RaisrFF_filter_path
		self.test_RaisrFF_scale_interpolation      = parserargs.test_RaisrFF_scale_interpolation
		self.test_RaisrFF_bits                     = parserargs.test_RaisrFF_bits
		self.test_RaisrFF_range                    = parserargs.test_RaisrFF_range
		self.test_RaisrFF_threadcount              = parserargs.test_RaisrFF_threadcount
		self.test_RaisrFF_blending			       = parserargs.test_RaisrFF_blending
		self.test_RaisrFF_passes                   = parserargs.test_RaisrFF_passes
		self.test_RaisrFF_mode                     = parserargs.test_RaisrFF_mode
		self.test_upscale_factor                   = parserargs.test_upscale_factor
		self.test_name                             = parserargs.test_name
		if self.test_name == "":
			self.test_name = self.test_algorithm.lower()
		self.test_tile                             = parserargs.test_tile
		self.test_tile_division                    = parserargs.test_tile_division
		self.test_tile_overlap                     = parserargs.test_tile_overlap
		self.test_video_create                     = parserargs.test_video_create
		self.test_video_quality                    = parserargs.test_video_quality

		self.metrics_skip_stage                    = parserargs.metrics_skip_stage
		self.metrics_overwrite                     = parserargs.metrics_overwrite
		self.metrics_psnr                          = parserargs.metrics_psnr
		self.metrics_ssim                          = parserargs.metrics_ssim
		self.metrics_msssim                        = parserargs.metrics_msssim
		self.metrics_vmaf                          = parserargs.metrics_vmaf
		self.metrics_vmaf_preload                  = parserargs.metrics_vmaf_preload
		self.metrics_vmaf_delete_yuvs              = parserargs.metrics_vmaf_delete_yuvs
		self.metrics_single_frame                  = parserargs.metrics_single_frame
		self.metrics_gmaf                          = parserargs.metrics_gmaf
		self.metrics_gmaf_delete_yuvs              = parserargs.metrics_gmaf_delete_yuvs
		self.metrics_lpips                         = parserargs.metrics_lpips
		self.metrics_haarpsi                       = parserargs.metrics_haarpsi
		self.metrics_test_name                     = parserargs.metrics_test_name
		self.metrics_border_crop                   = parserargs.metrics_border_crop

		self.log_skip_stage                        = parserargs.log_skip_stage
		self.log_logfilename                       = parserargs.log_logfilename
		self.log_overwrite                         = parserargs.log_overwrite
		self.log_test_name                         = parserargs.log_test_name
		if self.log_test_name == "":
			self.log_test_name = self.metrics_test_name

		self.compare_skip_stage                    = parserargs.compare_skip_stage
		self.compare_overwrite                     = parserargs.compare_overwrite
		self.compare_visualqualitytool             = parserargs.compare_visualqualitytool
		self.compare_original                      = parserargs.compare_original
		self.compare_nearest_neighbor              = parserargs.compare_nearest_neighbor
		self.compare_bilinear                      = parserargs.compare_bilinear
		self.compare_area                          = parserargs.compare_area
		self.compare_bicubic                       = parserargs.compare_bicubic
		self.compare_lanczos                       = parserargs.compare_lanczos
		self.compare_tests                         = parserargs.compare_tests
		self.compare_roifilename                   = parserargs.compare_roifilename
		self.compare_descriptions                  = parserargs.compare_descriptions
		self.compare_numcolumns                    = parserargs.compare_numcolumns

		self.plot_skip_stage                       = parserargs.plot_skip_stage
		self.plot_test_name                        = parserargs.plot_test_name
		if len(parserargs.plot_metrics) > 0:
			self.plot_metrics                      = [item for item in parserargs.plot_metrics.split(',')]

		self.absolute_project_folder               = ""
		self.absolute_videos_folder                = ""
		self.absolute_HR_folder                    = ""
		self.absolute_LR_folder                    = ""
		self.absolute_CHR_folder                   = ""
		self.absolute_test_folder                  = ""
		self.absolute_train_folder                 = ""
		self.absolute_contact_sheets_folder        = ""
		self.absolute_models_folder                = ""
		self.absolute_compare_folder               = ""
		self.absolute_vqt_folder                   = ""

set = Settings()