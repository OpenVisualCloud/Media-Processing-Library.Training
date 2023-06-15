# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
from stage import Stage, StageEnum
import settings
import os
import database
import csv

STAGE_NAME = 'StageLog'

class StageLog(Stage):

    def __init__(self):
        self.enum = StageEnum.LOG
        pass

    def CreateCompositeLog(self):

        # see if log file exists
        logfilepath = os.path.join(settings.set.absolute_project_folder, settings.set.log_logfilename)
        mode = 'w'
        if os.path.exists(logfilepath):
            if settings.set.log_overwrite:
                logging.info("StageLog..Log file exists, overwriting: " + logfilepath)
                mode = 'w'
            else:
                logging.info("StageLog..Log file exists, appending: " + logfilepath)
                mode = 'a'
        else:
            logging.info("StageLog..Creating new log file: " + logfilepath)
            mode = 'w'

        try:
            with open(logfilepath, mode=mode, newline='') as csv_file:
                logging.info("StageLog..log file opening for writing " + logfilepath)
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                if mode == 'w':
 
                    csv_writer.writerow(
                        ["video",
                         "folder",
                         "scene_index",
                         "start_frame",
                         "end_frame",
                         "num_frames",
                         "frame_width",
                         "frame_height",
                         "cropped",
                         "cropped_num_frames",
                         "cropped_frame_width",
                         "cropped_frame_height",
                         "classification", "algorithm",
                         "psnr_yuv",
                         "psnr_y",
                         "psnr_u",
                         "psnr_v",
                         "ssim_yuv",
                         "ssim_y",
                         "ssim_u",
                         "ssim_v",
                         "msssim",
                         "vmaf",
                         "vmaf_adm2",
                         "vmaf_adm_scale0","vmaf_adm_scale1","vmaf_adm_scale2", "vmaf_adm_scale3",
                         "vmaf_adm_motion2","vmaf_adm_motion",
                         "vmaf_vif_scale0","vmaf_vif_scale1","vmaf_vif_scale2","vmaf_vif_scale3",
                         "gmaf",
                         "gmaf_adm2",
                         "gmaf_adm_scale0", "gmaf_adm_scale1", "gmaf_adm_scale2", "gmaf_adm_scale3",
                         "gmaf_adm_motion2", "gmaf_adm_motion",
                         "gmaf_vif_scale0", "gmaf_vif_scale1", "gmaf_vif_scale2", "gmaf_vif_scale3",
                         "lpips",
                         "haarpsi",
                         "average test time"])

                sceneindices = database.db.getSceneIndices()

                for sceneindex in sceneindices:
                    scene = database.db.getScene(sceneindex)

                    if settings.set.log_test_name != "" and settings.set.log_test_name not in scene["tests"]:
                        logging.warning("WARNING: test not found for scene " + str(scene['scene_index']) + " test: " + settings.set.log_test_name)
                        continue

                    for test_name in scene['tests']:

                        if settings.set.log_test_name != "" and test_name != settings.set.log_test_name:
                            continue

                        # initial values to be written
                        psnr_yuv = -1
                        psnr_y = -1
                        psnr_u = -1
                        psnr_v = -1
                        ssim_yuv = -1
                        ssim_y = -1
                        ssim_u = -1
                        ssim_v = -1
                        msssim = -1
                        vmaf = -1
                        vmaf_adm2 = -1
                        vmaf_adm_scale0 = -1
                        vmaf_adm_scale1 = -1
                        vmaf_adm_scale2 = -1
                        vmaf_adm_scale3 = -1
                        vmaf_adm_motion2 = -1
                        vmaf_adm_motion = -1
                        vmaf_vif_scale0 = -1
                        vmaf_vif_scale1 = -1
                        vmaf_vif_scale2 = -1
                        vmaf_vif_scale3 = -1

                        gmaf = -1
                        gmaf_adm2 = -1
                        gmaf_adm_scale0 = -1
                        gmaf_adm_scale1 = -1
                        gmaf_adm_scale2 = -1
                        gmaf_adm_scale3 = -1
                        gmaf_adm_motion2 = -1
                        gmaf_adm_motion = -1
                        gmaf_vif_scale0 = -1
                        gmaf_vif_scale1 = -1
                        gmaf_vif_scale2 = -1
                        gmaf_vif_scale3 = -1

                        lpips = -1
                        haarpsi = -1

                        average_test_time = -1

                        metrics_dict = scene["tests"][test_name]
                        if 'psnr_yuv' in metrics_dict:
                            psnr_yuv = metrics_dict['psnr_yuv']
                        if 'psnr_y' in metrics_dict:
                            psnr_y = metrics_dict['psnr_y']
                        if 'psnr_u' in metrics_dict:
                            psnr_u = metrics_dict['psnr_u']
                        if 'psnr_v' in metrics_dict:
                            psnr_v = metrics_dict['psnr_v']

                        if 'ssim_yuv' in metrics_dict:
                            ssim_yuv = metrics_dict['ssim_yuv']
                        if 'ssim_y' in metrics_dict:
                            ssim_y = metrics_dict['ssim_y']
                        if 'ssim_u' in metrics_dict:
                            ssim_u = metrics_dict['ssim_u']
                        if 'ssim_v' in metrics_dict:
                            ssim_v = metrics_dict['ssim_v']

                        if 'msssim' in metrics_dict:
                            msssim = metrics_dict['msssim']

                        if 'vmaf' in metrics_dict:
                            vmaf = metrics_dict['vmaf']
                        if 'vmaf_adm2' in metrics_dict:
                            vmaf_adm2 = metrics_dict['vmaf_adm2']
                        if 'vmaf_adm_scale0' in metrics_dict:
                            vmaf_adm_scale0 = metrics_dict['vmaf_adm_scale0']
                        if 'vmaf_adm_scale1' in metrics_dict:
                            vmaf_adm_scale1 = metrics_dict['vmaf_adm_scale1']
                        if 'vmaf_adm_scale2' in metrics_dict:
                            vmaf_adm_scale2 = metrics_dict['vmaf_adm_scale2']
                        if 'vmaf_adm_scale3' in metrics_dict:
                            vmaf_adm_scale3 = metrics_dict['vmaf_adm_scale3']
                        if 'vmaf_adm_motion2' in metrics_dict:
                            vmaf_adm_motion2 = metrics_dict['vmaf_adm_motion2']
                        if 'vmaf_adm_motion' in metrics_dict:
                            vmaf_adm_motion = metrics_dict['vmaf_adm_motion']
                        if 'vmaf_vif_scale0' in metrics_dict:
                            vmaf_vif_scale0 = metrics_dict['vmaf_vif_scale0']
                        if 'vmaf_vif_scale1' in metrics_dict:
                            vmaf_vif_scale1 = metrics_dict['vmaf_vif_scale1']
                        if 'vmaf_vif_scale2' in metrics_dict:
                            vmaf_vif_scale2 = metrics_dict['vmaf_vif_scale2']
                        if 'vmaf_vif_scale3' in metrics_dict:
                            vmaf_vif_scale3 = metrics_dict['vmaf_vif_scale3']

                        if 'gmaf' in metrics_dict:
                            gmaf = metrics_dict['gmaf']
                        if 'gmaf_adm2' in metrics_dict:
                            gmaf_adm2 = metrics_dict['gmaf_adm2']
                        if 'gmaf_adm_scale0' in metrics_dict:
                            gmaf_adm_scale0 = metrics_dict['gmaf_adm_scale0']
                        if 'gmaf_adm_scale1' in metrics_dict:
                            gmaf_adm_scale1 = metrics_dict['gmaf_adm_scale1']
                        if 'gmaf_adm_scale2' in metrics_dict:
                            gmaf_adm_scale2 = metrics_dict['gmaf_adm_scale2']
                        if 'gmaf_adm_scale3' in metrics_dict:
                            gmaf_adm_scale3 = metrics_dict['gmaf_adm_scale3']
                        if 'gmaf_adm_motion2' in metrics_dict:
                            gmaf_adm_motion2 = metrics_dict['gmaf_adm_motion2']
                        if 'gmaf_adm_motion' in metrics_dict:
                            gmaf_adm_motion = metrics_dict['gmaf_adm_motion']
                        if 'gmaf_vif_scale0' in metrics_dict:
                            gmaf_vif_scale0 = metrics_dict['gmaf_vif_scale0']
                        if 'gmaf_vif_scale1' in metrics_dict:
                            gmaf_vif_scale1 = metrics_dict['gmaf_vif_scale1']
                        if 'gmaf_vif_scale2' in metrics_dict:
                            gmaf_vif_scale2 = metrics_dict['gmaf_vif_scale2']
                        if 'gmaf_vif_scale3' in metrics_dict:
                            gmaf_vif_scale3 = metrics_dict['gmaf_vif_scale3']

                        if 'lpips' in metrics_dict:
                            lpips = metrics_dict['lpips']

                        if 'haarpsi' in metrics_dict:
                            haarpsi = metrics_dict['haarpsi']

                        if 'average_test_time' in metrics_dict:
                            average_test_time = metrics_dict['average_test_time']

                        csv_writer.writerow([scene['video_name'],
                                             scene['folder_name'],
                                             scene['scene_index'],
                                             scene['start_frame'],
                                             scene['end_frame'],
                                             scene['num_frames'],
                                             scene['frame_width'],
                                             scene['frame_height'],
                                             scene['cropped'],
                                             scene['cropped_num_frames'],
                                             scene['cropped_frame_width'],
                                             scene['cropped_frame_height'],
                                             scene['classification'],
                                             test_name,
                                             '{:4f}'.format(psnr_yuv),
                                             '{:4f}'.format(psnr_y),
                                             '{:4f}'.format(psnr_u),
                                             '{:4f}'.format(psnr_v),

                                             '{:4f}'.format(ssim_yuv),
                                             '{:4f}'.format(ssim_y),
                                             '{:4f}'.format(ssim_u),
                                             '{:4f}'.format(ssim_v),

                                             '{:4f}'.format(msssim),

                                             '{:4f}'.format(vmaf),
                                             '{:4f}'.format(vmaf_adm2),
                                             '{:4f}'.format(vmaf_adm_scale0),
                                             '{:4f}'.format(vmaf_adm_scale1),
                                             '{:4f}'.format(vmaf_adm_scale2),
                                             '{:4f}'.format(vmaf_adm_scale3),
                                             '{:4f}'.format(vmaf_adm_motion2),
                                             '{:4f}'.format(vmaf_adm_motion),
                                             '{:4f}'.format(vmaf_vif_scale0),
                                             '{:4f}'.format(vmaf_vif_scale1),
                                             '{:4f}'.format(vmaf_vif_scale2),
                                             '{:4f}'.format(vmaf_vif_scale3),

                                             '{:4f}'.format(gmaf),
                                             '{:4f}'.format(gmaf_adm2),
                                             '{:4f}'.format(gmaf_adm_scale0),
                                             '{:4f}'.format(gmaf_adm_scale1),
                                             '{:4f}'.format(gmaf_adm_scale2),
                                             '{:4f}'.format(gmaf_adm_scale3),
                                             '{:4f}'.format(gmaf_adm_motion2),
                                             '{:4f}'.format(gmaf_adm_motion),
                                             '{:4f}'.format(gmaf_vif_scale0),
                                             '{:4f}'.format(gmaf_vif_scale1),
                                             '{:4f}'.format(gmaf_vif_scale2),
                                             '{:4f}'.format(gmaf_vif_scale3),

                                             '{:4f}'.format(lpips),

                                             '{:4f}'.format(haarpsi),
                                             '{:4f}'.format(average_test_time)])

        except Exception:
            logging.warning("WARNING - unable to open or write to log file: " + logfilepath)

    def CreateTestSceneLog(self, scene, test_name):
        import json

        # see if log file exists
        logfilepath = os.path.join(settings.set.absolute_test_folder, test_name, scene['folder_name'],"frames.csv")
        mode = 'w'
        if os.path.exists(logfilepath):
            if settings.set.log_overwrite:
                logging.info("StageLog..Test Scene Log file exists, overwriting: " + logfilepath)
                mode = 'w'
            else:
                logging.info("StageLog..Test Scene Log file exists, appending: " + logfilepath)
                mode = 'a'
        else:
            logging.info("StageLog..Creating new Test Scene Log file: " + logfilepath)
            mode = 'w'


        metricsfilepath = os.path.join(settings.set.absolute_test_folder, test_name, scene['folder_name'],"metrics.json")
        if not os.path.exists(metricsfilepath):
            logging.warning("StageLog..Unable to open metrics file: " + metricsfilepath + " for scene: " + scene['folder_name'] + " test:"+ test_name)
            return

        numframes = scene['num_frames']
        with open( metricsfilepath, 'r') as f:
            metrics_data = json.load(f)
            if 'psnr_yuvs' in metrics_data:
                psnr_yuvs = metrics_data["psnr_yuvs"]
            else:
                psnr_yuvs = [0] * numframes

            if 'psnr_ys' in metrics_data:
                psnr_ys = metrics_data["psnr_ys"]
            else:
                psnr_ys = [0] * numframes

            if 'psnr_us' in metrics_data:
                psnr_us   = metrics_data["psnr_us"]
            else:
                psnr_us = [0] * numframes

            if 'psnr_vs' in metrics_data:
                psnr_vs   = metrics_data["psnr_vs"]
            else:
                psnr_vs = [0] * numframes

            if 'ssim_yuvs' in metrics_data:
                ssim_yuvs = metrics_data["ssim_yuvs"]
            else:
                ssim_yuvs = [0] * numframes
            if 'ssim_ys' in metrics_data:
                ssim_ys = metrics_data["ssim_ys"]
            else:
                ssim_ys = [0] * numframes
            if 'ssim_us' in metrics_data:
                ssim_us = metrics_data["ssim_us"]
            else:
                ssim_us = [0] * numframes
            if 'ssim_vs' in metrics_data:
                ssim_vs = metrics_data["ssim_vs"]
            else:
                ssim_vs = [0] * numframes

            if 'mssims' in metrics_data:
                msssims   = metrics_data["msssims"]
            else:
                msssims = [0] * numframes

            if 'vmafs' in metrics_data:
                vmafs = metrics_data["vmafs"]
            else:
                vmafs = [0] * numframes

            if 'gmafs' in metrics_data:
                gmafs = metrics_data["gmafs"]
            else:
                gmafs = [0] * numframes

            if 'lpipss' in metrics_data:
                lpipss = metrics_data["lpipss"]
            else:
                lpipss = [0] * numframes

            if 'haarpsis' in metrics_data:
                haarpsis  = metrics_data["haarpsis"]
            else:
                haarpsis = [0] * numframes
        try:
            with open(logfilepath, mode=mode, newline='') as csv_file:
                logging.info("StageLog..log file opening for writing " + logfilepath)
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                if mode == 'w':

                    csv_writer.writerow(["Frame", "psnr_yuv", "psnr_y", "psnr_u", "psnr_v", "ssim_yuv","ssim_y", "ssim_u", "ssim_v", "mssim", "vmaf", "gmaf", "lpips", "haarpsi"])
                    for frame in range(numframes):
                        csv_writer.writerow([str(frame),
                                             '{:4f}'.format(psnr_yuvs[frame]),
                                             '{:4f}'.format(psnr_ys[frame]),
                                             '{:4f}'.format(psnr_us[frame]),
                                             '{:4f}'.format(psnr_vs[frame]),
                                             '{:4f}'.format(ssim_yuvs[frame]),
                                             '{:4f}'.format(ssim_ys[frame]),
                                             '{:4f}'.format(ssim_us[frame]),
                                             '{:4f}'.format(ssim_vs[frame]),
                                             '{:4f}'.format(msssims[frame]),
                                             '{:4f}'.format(vmafs[frame]),
                                             '{:4f}'.format(gmafs[frame]),
                                             '{:4f}'.format(lpipss[frame]),
                                             '{:4f}'.format(haarpsis[frame])])

        except Exception:
            logging.warning("WARNING - unable to open or write to log file: " + logfilepath)

    def ExecuteStage(self):
        logging.info("StageLog..executing stage")

        if settings.set.log_skip_stage:
            logging.info("StageLog..skipping stage")
            return

        self.CreateCompositeLog()

        # for each scene
        sceneindices = database.db.getSceneIndices()
        for sceneid in sceneindices:
            scene = database.db.getScene(sceneid)

            # for each test recorded for the scene
            for test_name in scene['tests']:
                self.CreateTestSceneLog(scene, test_name)

        logging.info("StageLog..stage finished")