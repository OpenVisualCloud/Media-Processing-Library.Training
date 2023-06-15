# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
import database
import os
import glob
from stage import Stage, StageEnum
from PIL import Image, ImageDraw, ImageFont
import csv
import re
import multiprocessing as mp
import copy
import shutil
import json
import utils

STAGE_NAME = 'StageCompare'

class StageCompare(Stage):

    def __init__(self):
        self.enum = StageEnum.COMPARE
        pass

    def make_roi_source(self, sourceim, X, Y, width, height):
        WHITE_COLOR = (255, 255, 255)
        RED_COLOR = (255,0,0)

        # get the size of the source image
        isize = (sourceim.shape[1], sourceim.shape[0])

        # create a new image of the same size
        inew = Image.new('RGB', isize, WHITE_COLOR)

        # draw the source image
        bbox = (0,0, sourceim.shape[1], sourceim.shape[0])
        inew.paste(Image.fromarray(sourceim), bbox)

        # create a draw object to draw lines on
        draw = ImageDraw.Draw(inew)

        # red rectangle for the roi
        draw.rectangle([X, Y, X+width-1, Y+height-1], outline=RED_COLOR, width=3)

        # return the image
        return inew

    def get_text_size(self, text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        return draw.textsize(text, font)

    def find_font_size(self, text, font, image, target_width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = self.get_text_size(text, image, tested_font)
        estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
        return round(estimated_font_size)

    # adapted from https://code.activestate.com/recipes/412982-use-pil-to-make-a-contact-sheet-montage-of-images/
    def make_compare_sheet(self, title, images, image_width, image_height, show_descriptions, num_columns):

        margin_left  = 20
        margin_right = 20
        margin_top = 20
        margin_bottom = 20
        padding = 10
        if num_columns == -1:
            ncols = len(images)
        else:
            ncols = num_columns
        nrows = 1 + int((len(images)-1)/ncols)

        NUM_DESCRIPTION_LINES = 1
        WHITE_COLOR = (255, 255, 255)
        BLACK_COLOR = (20,0,0)
        BLUE_COLOR = (30,144,255)
        ORANGE_COLOR = (255, 165, 0)

        # title appears at the top of image
        TITLEHEIGHT = 30

        # descriptions appear below the photos
        if show_descriptions:
            DESCRIPTION_HEIGHT = 24
        else:
            DESCRIPTION_HEIGHT = 0

        # compute various dimensions (margins, etc)
        marw = margin_left + margin_right
        marh = margin_top + margin_bottom
        padw = (ncols - 1) * padding
        padh = (nrows - 1) * padding

        # determine overall width/height of the image
        imagewidth = ncols * image_width + marw + padw
        imageheight = nrows * (image_height + DESCRIPTION_HEIGHT) + marh + padh + TITLEHEIGHT
        isize = (imagewidth, imageheight)

        # Create the new image. The background doesn't have to be white
        inew = Image.new('RGB', isize, WHITE_COLOR)

        # create the fonts.  Will use FreeSans.ttf in fonts-freefont-ttf
        font_path = os.path.join("FAIME","resources","FreeSans.ttf")

        text = " " * 75 * ncols
        fontsmall_size = int(self.find_font_size(text, font_path, inew, 1))
        fontsmall = ImageFont.truetype(font_path, fontsmall_size)

        fontlarge_size = int(fontsmall_size*1.4)
        fontlarge = ImageFont.truetype(font_path, fontlarge_size)
        # draw lines and text for each image
        imageindex = 0
        numimages = len(images)

        # create a draw object to draw text and lines on
        draw = ImageDraw.Draw(inew)

        # draw title
        sizew, sizeh = draw.textsize(title, fontlarge)
        draw.text((int(imagewidth / 2 - sizew / 2), int(TITLEHEIGHT / 2 - sizeh / 2)), title, (0, 0, 0), font=fontlarge,
                  align="ms")

        for irow in range(nrows):
            for icol in range(ncols):
                if imageindex < numimages:
                    left = margin_left + icol * (image_width + padding)
                    right = left + image_width
                    upper = TITLEHEIGHT + margin_top + irow * (
                                image_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
                    lower = upper + image_height
                    bbox = (left, upper, right, lower)
                    img = images[imageindex]['image']

                    # draw the image
                    inew.paste(Image.fromarray(img), bbox)

                    # black rectangle for the description block
                    draw.rectangle([left,lower,right-1,lower+DESCRIPTION_HEIGHT], fill=BLACK_COLOR)

                    # Description
                    if show_descriptions:
                        center = int((left + right) / 2)
                        middle = int(lower + DESCRIPTION_HEIGHT / 2)
                        desc = images[imageindex]['description']
                        if len(images[imageindex]['metrics'])> 0:
                            desc += ' ' + images[imageindex]['metrics']
                        sizew, sizeh = draw.textsize(desc, fontsmall)
                        color = WHITE_COLOR
                        if images[imageindex]['type'] == 0:
                            color = WHITE_COLOR
                        elif images[imageindex]['type'] == 1:
                            color = BLUE_COLOR
                        elif images[imageindex]['type'] == 2:
                            color = ORANGE_COLOR
                        draw.text((int(center - sizew / 2), int(middle - sizeh / 2)), desc, color, font=fontsmall,
                              align="ms")

                imageindex = imageindex + 1

        # return the image
        return inew

    def ProcessROI(self, set, roiid, scene, foldername, startframe, endframe, X, Y, width, height):
        import cv2
        import numpy as np
        import imageio

        # Configure Logging
        logger = logging.getLogger(__name__)

        if scene == None:
            logging.info("WARNING: scene not found in database:"+foldername)
            return

        if startframe < 0 or startframe >= scene['num_frames']:
            logging.info("WARNING: invalid start frame in ROI: " + str(startframe) + " # of frames in scene=" + str(scene['num_frames']))
            return
        if endframe == -1:
            endframe = scene['num_frames']-1
        if endframe < 0 or endframe >= scene['num_frames'] or endframe <startframe:
            logging.info("WARNING: invalid end frame in ROI: " + str(endframe) + " # of frames in scene=" + str(scene['num_frames']) + " start frame="+ str(startframe))
            return

        if X< 0 or X >= scene['frame_width'] or Y <0 or Y >= scene['frame_height']:
            logging.info("WARNING: invalid region location in ROI: " + str(X) + ","  + str(Y) + " frames size=" + str(scene['frame_width']) + "," + str(scene['frame_height']))
            return

        if width < 0 or (X+width) > scene['frame_width'] or height < 0 or (Y+height) > scene['frame_height']:
            logging.info("WARNING: region not in frame: " + str(X) + ","  + str(Y) + ","  + str(width) +  ","  + str(height) + " frame size=" + str(scene['frame_width']) + "," + str(scene['frame_height']))
            return

        HR_width = scene['frame_width']
        HR_height = scene['frame_height']

        # build up a list of images to put on the compare sheet
        compare_images = []
        compare_sequences = []

        # get the HR frame:
        if set.compare_original:
            if scene['cropped']:
                HRfoldername = scene['cropped_path']
            else:
                HRfoldername = scene['hr_path']

            if not os.path.exists(HRfoldername):
                logging.warning("WARNING: HR folder for scene does not exist:" + HRfoldername)
                return

            # get a list of the HR files
            HRframes = utils.GetFiles(HRfoldername)
            for index, filepath in enumerate(HRframes):

                if index < startframe or index > endframe:
                    continue
                HRim: np.ndarray = cv2.imread(filepath)

                # swap red and blue channels
                HRim = cv2.cvtColor(HRim, cv2.COLOR_BGR2RGB)

                # crop
                HRim = HRim[Y:Y + height, X:X + width]

                if index == startframe:
                    compare_images.append({"description": "Ground_Truth", "metrics": "", "image": HRim, "type": 0})
                    compare_sequences.append({"description": "Ground_Truth", "metrics": "", "images": [HRim], "type":0})
                else:
                    compare_sequences[-1]["images"].append(HRim)

        # get a list of the LR files
        LRfoldername = scene['lr_path']
        if not os.path.exists(LRfoldername):
            logging.warning("WARNING: LR folder for scene does not exist:" + LRfoldername)
            return
        LRframes = utils.GetFiles(LRfoldername)

        for index, filepath in enumerate(LRframes):
            if index < startframe or  index > endframe:
                continue

            # use the first frame for the LR
            LRvideofilename = filepath
            LRim: np.ndarray = cv2.imread(LRvideofilename)

            # swap red and blue channels
            LRim = cv2.cvtColor(LRim, cv2.COLOR_BGR2RGB)

            if set.compare_nearest_neighbor:
                NNim = cv2.resize(LRim,
                                  dsize=(int(HR_width),
                                         int(HR_height)),
                                  interpolation=cv2.INTER_NEAREST)
                NNim = NNim[Y:Y + height, X:X + width]
                if index == startframe:
                    compare_images.append({"description": "Nearest", "metrics": "", "image": NNim, "type":1})
                    compare_sequences.append({"description": "Nearest", "metrics": "", "images": [NNim], "type":0})
                else:
                    for seq in compare_sequences:
                        if seq["description"] == "Nearest":
                            seq["images"].append(NNim)
                            break

            if set.compare_bilinear:
                BLim = cv2.resize(LRim,
                                  dsize=(int(HR_width),
                                         int(HR_height)),
                                  interpolation=cv2.INTER_LINEAR)
                BLim = BLim[Y:Y + height, X:X + width]
                if index == startframe:
                    compare_images.append({"description": "Bilinear", "metrics": "", "image": BLim, "type":1})
                    compare_sequences.append({"description": "Bilinear", "metrics": "", "images": [BLim], "type": 0})
                else:
                    for seq in compare_sequences:
                        if seq["description"] == "Bilinear":
                            seq["images"].append(BLim)
                            break

            if set.compare_area:
                Areaim = cv2.resize(LRim,
                                  dsize=(int(HR_width),
                                         int(HR_height)),
                                  interpolation=cv2.INTER_AREA)
                Areaim = Areaim[Y:Y + height, X:X + width]
                if index == startframe:
                    compare_images.append({"description": "Area", "metrics": "", "image": Areaim, "type":1})
                    compare_sequences.append({"description": "Area", "metrics": "", "images": [Areaim], "type": 0})
                else:
                    for seq in compare_sequences:
                        if seq["description"] == "Area":
                            seq["images"].append(Areaim)
                            break

            if set.compare_bicubic:
                CBim = cv2.resize(LRim,
                                  dsize=(int(HR_width),
                                         int(HR_height)),
                                    interpolation=cv2.INTER_CUBIC)
                CBim = CBim[Y:Y + height, X:X + width]
                if index == startframe:
                    compare_images.append({"description": "Bicubic", "metrics": "", "image": CBim, "type":1})
                    compare_sequences.append({"description": "Bicubic", "metrics": "", "images": [CBim], "type": 0})
                else:
                    for seq in compare_sequences:
                        if seq["description"] == "Bicubic":
                            seq["images"].append(CBim)
                            break

            if set.compare_lanczos:
                LCim = cv2.resize(LRim,
                                  dsize=(int(HR_width),
                                         int(HR_height)),
                                  interpolation=cv2.INTER_LANCZOS4)
                LCim = LCim[Y:Y + height, X:X + width]
                if index == startframe:
                    compare_images.append({"description": "Lanczos", "metrics": "", "image": LCim, "type":1})
                    compare_sequences.append({"description": "Lanczos", "metrics": "","images": [LCim], "type": 0})
                else:
                    for seq in compare_sequences:
                        if seq["description"] == "Lanczos":
                            seq["images"].append(LCim)
                            break

        # go through all of the scene tests
        if set.compare_tests:
            for test_name in scene['tests']:
                test_folder = scene['tests'][test_name]['test_path']

                if not os.path.exists(test_folder):
                    logging.warning("WARNING test folder does not exist for test: " + test_folder)
                    continue

                testfiles = utils.GetFiles(test_folder)
                if len(testfiles)< startframe:
                    logging.warning("WARNING: Test file not exist for frame:" + str(startframe) + "in folder: " + str(test_folder))
                    logging.warning("WARNING: Test file not exist for frame:" + str(startframe) + "in folder: " + str(test_folder))
                    continue

                for index, filepath in enumerate(testfiles):
                    if index < startframe or index > endframe:
                        continue

                    Testim: np.ndarray = cv2.imread(testfiles[index])

                    # swap red and blue channels
                    Testim = cv2.cvtColor(Testim, cv2.COLOR_BGR2RGB)

                    # crop
                    Testim = Testim[Y:Y + height, X:X + width]
                    if index == startframe:
                        metrics = ""
                        metrics_dict = scene["tests"][test_name]
                        if 'psnr_yuv' in metrics_dict:
                            psnr_yuv = metrics_dict['psnr_yuv']
                            if psnr_yuv > 0.0:
                                metrics += ' PNSR:' + '{:6.2f}'.format(psnr_yuv)
                        if 'vmaf' in metrics_dict:
                            vmaf = metrics_dict['vmaf']
                            if vmaf > 0.0:
                                metrics += ' VMAF:' + '{:6.2f}'.format(vmaf)
                        compare_images.append({"description": test_name, "metrics":metrics, "image": Testim, "type":2})
                        compare_sequences.append({"description": test_name, "metrics":metrics, "images": [Testim], "type": 0})
                    else:
                        for seq in compare_sequences:
                            if seq["description"] == test_name:
                                seq["images"].append(Testim)
                                break

        # construct the roi folder
        roifolder = os.path.join(set.absolute_compare_folder, "roi%004d_%s" % (roiid, foldername))
        if not os.path.exists(roifolder):
            os.makedirs(roifolder)

        # construct a title for the comparision sheet
        title = "ROI# " + str(roiid) + foldername + " (frame "+str(startframe) + ") from " + scene['video_name']  + " ["+ str(X)+","+str(Y) + "," + str(width) + "," + str(height) + "]"

        # Check if the HR folder exists
        HRfoldername = scene['hr_path']
        if not os.path.exists(HRfoldername):
            logging.warning("WARNING: HR folder for scene does not exist:" + HRfoldername)
            return

        # get a list of the HR files
        HRframes = utils.GetFiles(HRfoldername)
        sourceim: np.ndarray = cv2.imread(HRframes[startframe])
        # swap red and blue channels
        sourceim = cv2.cvtColor(sourceim, cv2.COLOR_BGR2RGB)
        roisourceimg = self.make_roi_source(sourceim, X,Y,width,height)

        # save the roi source image
        roisourcefilename = os.path.join(roifolder, "%s_source_X%d_Y%04d_W%d_H%d.png" % (foldername, X,Y,width,height))
        roisourceimg.save(roisourcefilename)

        # create the comparision sheet image
        comparision_sheet_img = self.make_compare_sheet(title, compare_images, width, height, set.compare_descriptions, set.compare_numcolumns)

        # save the comparison sheet image
        filename = os.path.join(roifolder, "%s_X%d_Y%04d_W%d_H%d.png" % (foldername, X,Y,width,height))
        comparision_sheet_img .save(filename)

        # save the animated gifs for the scene
        giffilenames = []
        descriptions = []
        metrics = []
        for seq in compare_sequences:
            giffilename = os.path.join(roifolder, "%s_%s_X%d_Y%04d_W%d_H%d.gif" % (foldername, seq["description"], X,Y,width,height))
            giffilenames.append(giffilename)
            descriptions.append(seq["description"])
            metrics.append(seq["metrics"])
            imageio.mimsave(giffilename, seq["images"])

        # save an html table to view the animated gifs:
        html_openbodytext = "<html>\n<body>\n"
        html_closebodytext = "</body>\n</html>\n"
        filename = os.path.join(roifolder, "compare_roi_%s_X%d_Y%04d_W%d_H%d.html" % (foldername, X,Y,width,height))
        file = open(filename, "w")
        file.write(html_openbodytext)

        file.write("<h1> Region of Interest Scene %s (X:%d Y:%d W:%d H:%d #%d-#%d)  </h1>\n" % (foldername, X,Y,width,height, startframe, endframe))

        file.write("<table style = ""width:100%"" >\n")

        num_columns = set.compare_numcolumns
        if num_columns == -1:
            ncols = len(compare_sequences)
        else:
            ncols = num_columns

        numSections = int((len(compare_sequences) + ncols-1)/ncols)
        percent = int(100/ncols)

        for sect in range(numSections):

            file.write("<tr>\n")

            for i in range(ncols):
                s = sect * ncols + i

                if s < len(compare_sequences):
                    seq = compare_sequences[s]
                    file.write("<th> %s </th>\n" % seq["description"])
                else:
                    file.write("<th> </th>\n")
            file.write("</tr>\n")

            file.write("<tr>\n")
            for i in range(ncols):
                s = sect * ncols + i
                if s < len(compare_sequences):
                    seq = compare_sequences[s]
                    giffilename = os.path.join("%s_%s_X%d_Y%04d_W%d_H%d.gif" % (foldername, seq["description"], X, Y, width, height))
                    file.write("<td width=""%d%%"" style=""text-align:center"";min-height:200px> <img src=%s width=""100%%""> </td>\n" % (percent, giffilename))
                else:
                    file.write("<td> </td>\n")
            file.write("</tr>\n")

        file.write("</table>\n")
        file.write(html_closebodytext)
        file.close()

        # write json file with all of the roi compare data
        compare_data = {}
        compare_data.update({
            'foldername': foldername,
            'descriptions': descriptions,
            'metrics': metrics,
            'giffilenames' : giffilenames,
            'roisourcefilename' : roisourcefilename,
            'X': X,
            'Y': Y,
            'width': width,
            'height': height,
            'startframe' : startframe,
            'endframe' : endframe
                })
        compare_path = os.path.join(roifolder, "compare.json")
        with open(compare_path, 'w') as f:
            json.dump(compare_data, f, indent=2, separators=(',', ':'))

        return

    def skip_comments(self, lines):
        """
        A filter which skip/strip the comments and yield the
        rest of the lines

        :param lines: any object which we can iterate through such as a file
            object, list, tuple, or generator
        """
        comment_pattern = re.compile(r'\s*#.*$')

        for line in lines:
            line = re.sub(comment_pattern, '', line).strip()
            if line:
                yield line

    # https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth/31039095
    def copytree(self, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.rmtree(d, ignore_errors=True, onerror=None)
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def CreateVQTConfig(self, relativepaths: bool=True):

        # get a list of scenes to process
        sceneindices = database.db.getSceneIndices()
        if len(sceneindices) == 0:
            logging.warning("StageCompare..WARNING no scenes to process")
            return

        # create vqt folder
        if not os.path.exists(settings.set.absolute_vqt_folder):
            os.mkdir(settings.set.absolute_vqt_folder)

        # Write out a configuration file for the Visual Quality Tool
        config_json = os.path.join(settings.set.absolute_vqt_folder, "vqt_config.json")

        cachejs = os.path.join(settings.set.absolute_vqt_folder, "cache.js")

        # open the file for writing
        config_file = open(config_json, 'w')

        # create  list of sequences (scenes)
        config = {'sequences': []}

        # for each scene to process
        for seq_idx, sceneindex in enumerate(sceneindices):

            # get the scene
            scene = database.db.getScene(sceneindex)

            # add a new sequence for the scene
            config['sequences'].append({
                'name': scene['folder_name'],
                'models': []
            })

            # set the first and last frame counts  for the sequence
            config['sequences'][seq_idx]['firstFrame'] = 0
            config['sequences'][seq_idx]['lastFrame'] = scene['num_frames']-1

            # Add ref/hr data
            config['sequences'][seq_idx]['ref'] = {
                'frames': [],
                'statistics': {},
                'video': ""
            }

            # Add the ref/hr png files
            if scene['cropped']:
                png_dir = scene['cropped_path']
            else:
                png_dir = scene['hr_path']

            png_files = utils.GetFiles(png_dir)
            for frame_idx, png_path in enumerate(png_files):
                if relativepaths:
                    png_path = os.path.relpath(png_path, settings.set.absolute_vqt_folder)
                config['sequences'][seq_idx]['ref']['frames'].append(png_path)

            # Add the ref/hr video
            video_path = os.path.join(settings.set.absolute_HR_folder, scene['folder_name'], "frames.mp4")
            if os.path.exists(video_path):
                if relativepaths:
                    video_path = os.path.relpath(video_path, settings.set.absolute_vqt_folder)
                config['sequences'][seq_idx]['ref']['video'] = video_path

            # for each test recorded for the scene, we'll add a model
            for model_idx, test_name in enumerate(scene['tests']):

                config['sequences'][seq_idx]['models'].append({
                    'name': test_name,
                    'frames': [],
                    'diffs': [],
                    'heatmaps': {},
                    'metrics': {},
                    'statistics': {},
                    'video': ""
                })

                # get the video for the test
                video_path = os.path.join(settings.set.absolute_test_folder, test_name, scene['folder_name'], "frames.mp4")
                if os.path.exists(video_path):
                    if relativepaths:
                        video_path = os.path.relpath(video_path, settings.set.absolute_vqt_folder)
                    config['sequences'][seq_idx]['models'][model_idx]['video'] = video_path

                # add the test pngs
                png_dir = os.path.join(settings.set.absolute_test_folder, test_name, scene['folder_name'])
                png_files = utils.GetFiles(png_dir)
                for frame_idx, png_path in enumerate(png_files):
                    if relativepaths:
                        png_path = os.path.relpath(png_path, settings.set.absolute_vqt_folder)
                    config['sequences'][seq_idx]['models'][model_idx]['frames'].append(png_path)

                # get the metrics dictionary for the test
                metrics_dict = scene["tests"][test_name]

                # define the metrics
                metrics       = ['psnr_yuv',  'psnr_y',  'psnr_u',  'psnr_v',  'ssim_yuv',  'msssim',  'haarpsi',  'vmaf',  'gmaf',  'lpips']
                metric_scores = ['psnr_yuvs', 'psnr_ys', 'psnr_us', 'psnr_vs', 'ssim_yuvs', 'msssims', 'haarpsis', 'vmafs', 'gmafs', 'lpipss']

                # for each possible metric
                for metric_idx, metric in enumerate(metrics):
                    # see if the metric is in the dictionary
                    if metric in metrics_dict:
                        # add it to the metrics dictionary
                        config['sequences'][seq_idx]['models'][model_idx]['metrics'][metric] = metrics_dict[metric_scores[metric_idx]]

                # get the difference image folder
                diff_dir = os.path.join(settings.set.absolute_test_folder, test_name, scene['folder_name'],"diffs")
                diff_files = utils.GetFiles(diff_dir)

                # add the difference images
                for frame_idx, diff_path in enumerate(diff_files):
                    if relativepaths:
                        diff_path = os.path.relpath(diff_path, settings.set.absolute_vqt_folder)
                    config['sequences'][seq_idx]['models'][model_idx]['diffs'].append(diff_path)

        # write out the config file
        json.dump(config, config_file, indent=4, sort_keys=True)

        # write the cache file with the the contents of the config file
        cachestr = "var cache_config = " + json.dumps(config, indent=4, sort_keys=True)
        cache_file = open(cachejs, "w")
        cache_file.write(cachestr)
        cache_file.close()

        # copy the vqt files for both relative and absolute paths
        vqt_src_path = os.path.join(os.getcwd(),"FAIME","visual_quality_tool")
        self.copytree(vqt_src_path, settings.set.absolute_vqt_folder)

    def ExecuteStage(self):
        logging.info("StageCompare..executing stage")

        if settings.set.compare_skip_stage:
            logging.info("StageCompare..skipping stage")
            return

        if settings.set.compare_visualqualitytool.lower() == "relative":
            self.CreateVQTConfig(True)
        elif settings.set.compare_visualqualitytool.lower() == "absolute":
            self.CreateVQTConfig(False)
        elif not settings.set.compare_visualqualitytool.lower() == "none":
            logging.warning("WARNING..unknwon compare_visualqualitytool option: " + settings.set.compare_visualqualitytool)

        if settings.set.compare_roifilename == "":
            logging.info("StageCompare..ROI filename not set")
            return

        roifilename = os.path.join(settings.set.absolute_project_folder, settings.set.compare_roifilename)

        if not os.path.exists(roifilename):
            logging.warning("WARNING StageCompare..ROI file does not exist: "+ roifilename)
            return

        # construct the compare folder
        comparefolder = settings.set.absolute_compare_folder
        if os.path.exists(comparefolder):
            if not settings.set.compare_overwrite:
                logging.info("StageCompare..compare folder already exists: " + comparefolder)
                return
            else:
               shutil.rmtree(comparefolder, ignore_errors=True)
        os.makedirs(comparefolder)

        rois = []
        with open(roifilename, newline='') as csvfile:

            roireader = csv.DictReader(self.skip_comments(csvfile))
            roiindex=0
            for row in roireader:
                if len(row) != 7:
                    logging.info("StageCompare..ROI file contains invalid region of interest at line " + str(roiindex+1))
                    return
                scene = database.db.findSceneFromFolderName(row['folder'])
                rois.append([roiindex, scene, row['folder'], int(row['startframe']), int(row['endframe']), int(row['X']), int(row['Y']), int(row['width']), int(row['height'])])
                roiindex += 1
            if roiindex==0:
                logging.warning("WARNING: ROI file does not contain any region definitions (is header line missing?): " + roifilename)
                return
        logging.info("StageCompare.. processing "+ str(len(rois)) + " regions")
        if settings.set.multiprocess:
            processes = []
            setcopy = copy.deepcopy(settings.set)
            for roi in rois:
                p = mp.Process(target=self.ProcessROI, args=(setcopy, roi[0], roi[1], roi[2], roi[3], roi[4], roi[5],roi[6], roi[7], roi[8]))
                processes.append(p)

            [p.start() for p in processes]
            [p.join() for p in processes]

            # copy back
            settings.set = copy.deepcopy(setcopy)
        else:
            for roi in rois:
                self.ProcessROI(settings.set, roi[0], roi[1], roi[2], roi[3], roi[4], roi[5],roi[6], roi[7], roi[8])

        logging.info("StageCompare.. processing complete")
