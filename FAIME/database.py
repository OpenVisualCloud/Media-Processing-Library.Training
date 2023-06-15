# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import settings
import logging
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont
import math
import glob
import utils
import video
import shutil
class Database:
    test_template = {
        'test_name' : "",
        'test_path': "",
        'psnr_yuv' : 0.0,
        'psnr_y': 0.0,
        'psnr_u': 0.0,
        'psnr_v': 0.0,
        'ssim' : 0.0,
        'msssim':  0.0,
        'vmaf_adm2': 0.0,
        'vmaf_adm_scale0': 0.0,
        'vmaf_adm_scale1': 0.0,
        'vmaf_adm_scale2': 0.0,
        'vmaf_adm_scale3': 0.0,
        'vmaf_adm_motion2': 0.0,
        'vmaf_adm_motion': 0.0,
        'vmaf_vif_scale0': 0.0,
        'vmaf_vif_scale1': 0.0,
        'vmaf_vif_scale2': 0.0,
        'vmaf_vif_scale3': 0.0,
        'vmaf': 0.0,
        "lpips" : 0.0,
        'average_test_time': 0.0,
        }

    scenedict_template = {
        'video_name': "",
        'scene_index': 0,
        'folder_name':"",
        'hr_path':"",
        'lr_path':"",
        'chr_path':"",
        'test_path': "",
        'video_index': 0,
        'start_frame': 0,
        'start_framecode': 0,
        'end_frame': 0,
        'end_framecode': 0,
        'num_frames': 0,
        'frame_width': 0,
        'frame_height': 0,
        'bit_depth':0,
        'cropped': False,
        'cropped_num_frames': 0,
        'cropped_frame_width': 0,
        'cropped_frame_height': 0,
        'cropped_path': "",
        'same_size' : True,
        'same_bits': True,
        'scene_detect_downscale': 1,
        'fps' : 0,
        'duplicated_frames' : False,
        'classification' : 0,
        'delta_hsv' : 0,
        'delta_hue' : 0,
        'delta_sat' : 0,
        'delta_lum' : 0,
        'max_delta_hsv' : 0,
        'max_delta_hsv_frame' : 0,
        'tests' : {},
        'diff' : 0.0,
        'avgdiff' : 0.0,
        'motion' : 0.0,
        'texture_complexity' : 0.0,
    }

    def init(self):
        self.scenes = []
        self.database_filename = ''
        pass

    def __init__(self):
        self.scenes = []
        self.database_filename = ''
        pass

    def open(self):
        # determine the database filename
        self.database_filename = os.path.join(os.path.abspath(settings.set.project_folder), "database.json")

        if os.path.exists(self.database_filename):
            # logging.info("Opening database: " + self.database_filename)
            with open(self.database_filename, 'r') as f:
                self.scenes = json.load(f)

            # add some newer keys to maintain backwards compatiability
            for scene in self.scenes:
                if 'duplicated_frames' not in scene:
                    scene['duplicated_frames'] = False
            return True
        else:
            logging.warning("WARNING - database not found: " + self.database_filename)
            return False

    def build(self):
        logging.info("INFO - Building database  project folder="+settings.set.project_folder)

        if not os.path.exists(settings.set.project_folder):
            logging.warning("ERROR - project folder does not exist: " + settings.set.project_folder)
            exit()

        self.database_filename = os.path.join(os.path.abspath(settings.set.project_folder), "database.json")
        if os.path.exists(self.database_filename) and len(self.scenes)>0:
            logging.warning("WARNING - database already exists and will be modified: " + self.database_filename)

        # try reading in HR folders
        if not os.path.exists(settings.set.absolute_HR_folder):
            logging.warning("WARNING - no HR folder found: " + settings.set.absolute_HR_folder)
            return
        subfolders = sorted([f.path for f in os.scandir(settings.set.absolute_HR_folder) if f.is_dir()])
        if len(subfolders) == 0:
            logging.warning("WARNING - no HR folders found in: " + settings.set.absolute_HR_folder)
            return

        for folder in subfolders:
            logging.info("processing folder="+folder)
            scene = self.scenedict_template.copy()
            head_tail = os.path.split(folder)
            logging.info("INFO - discovered: "+folder)

            foldername =  head_tail[1]
            # is this  new scene?
            scenefound = False
            for scenesearch in self.scenes:
                if scenesearch['folder_name'] == foldername:
                    scenefound = True
                    break
            if scenefound:
                logging.info("INFO - scene " + foldername + " is already in database")
                continue
            logging.info("INFO - scene " + foldername + " will be added to database")

            scene['folder_name'] = foldername
            scene['hr_path'] = os.path.join(settings.set.absolute_HR_folder, head_tail[1])
            scene['lr_path'] = ""
            scene['chr_path'] = ""
            scene['tests'] = {}
            files = utils.GetFiles(os.path.join(scene['hr_path']))

            scene['num_frames'] = len(files)
            scene['same_size'] = True
            if len(files) > 0:
                # see if all files are the same size
                for ind, file in enumerate(files):
                    img = Image.open(file)
                    bits = video.Video.getBits(file, ignore_errors=True)
                    bit_depth = bits
                    if ind == 0:
                        initwidth  = img.width
                        initheight = img.height
                        initbits = bit_depth
                        scene['frame_width'] = img.width
                        scene['frame_height'] = img.height
                        scene['bit_depth'] = bit_depth
                    else:
                        if initwidth != img.width or initheight != img.height:
                            scene['same_size'] = False
                            scene['frame_width'] = -1
                            scene['frame_height'] = -1
                        if initbits != bit_depth:
                            scene['bit_depth'] = -1
                            scene['same_bits'] = False
            scene['cropped'] = False
            scene['cropped_frame_width'] = -1
            scene['cropped_frame_height'] = -1
            scene['cropped_num_frames'] = 0
            scene['cropped_path'] = ""
            cropfolder = os.path.join(scene['hr_path'], "Crop")
            if os.path.exists(cropfolder):
                scene['cropped_path'] = cropfolder
                scene['cropped'] = True
                files = utils.GetFiles( cropfolder)
                if len(files) > 0:
                    scene['cropped_num_frames'] = len(files)
                    # see if all files are the same size
                    for ind, file in enumerate(files):
                        img = Image.open(file)
                        if ind == 0:
                            initwidth = img.width
                            initheight = img.height
                            scene['cropped_frame_width'] = img.width
                            scene['cropped_frame_height'] = img.height
                        else:
                            if initwidth != img.width or initheight != img.height:
                                scene['cropped_frame_width'] = -1
                                scene['cropped_frame_height'] = -1
                                break

            self.add(scene, False)

        # try reading in LR folders
        if not os.path.exists(settings.set.absolute_LR_folder):
            logging.warning("WARNING - no LR folder found: " + settings.set.absolute_LR_folder)
        else:
            subfolders = sorted([f.path for f in os.scandir(settings.set.absolute_LR_folder) if f.is_dir()])
            if len(subfolders) == 0:
                logging.warning("WARNING - no LR folders found in: " + settings.set.absolute_LR_folder)
            else:
                for folder in subfolders:
                    head_tail = os.path.split(folder)

                    scenefound = False
                    for scene in self.scenes:
                        if scene['folder_name'] == head_tail[1]:
                            scene['lr_path'] = os.path.join(settings.set.absolute_LR_folder, head_tail[1])
                            scenefound = True
                            break
                    if not scenefound:
                        logging.warning("WARNING - no matching scene found for " + folder + " scenename="+head_tail[1])

        # try reading in CHR folders
        if not os.path.exists(settings.set.absolute_CHR_folder):
            logging.warning("WARNING - no CHR folder found: " + settings.set.absolute_CHR_folder)
        else:
            subfolders = sorted([f.path for f in os.scandir(settings.set.absolute_CHR_folder) if f.is_dir()])
            if len(subfolders) == 0:
                logging.warning("WARNING - no CHR folders found in: " + settings.set.absolute_CHR_folder)
            else:
                for folder in subfolders:
                    head_tail = os.path.split(folder)

                    scenefound = False
                    for scene in self.scenes:
                        if scene['folder_name'] == head_tail[1]:
                            scene['chr_path'] = os.path.join(settings.set.absolute_CHR_folder, head_tail[1])
                            scenefound = True
                            break
                    if not scenefound:
                        logging.warning("WARNING - no matching scene found for " + folder + " scenename="+head_tail[1])

        # try reading in test folders
        if not os.path.exists(settings.set.absolute_test_folder):
            logging.warning("WARNING - no test folder found: " + settings.set.absolute_test_folder)
        else:
            subfolders = sorted([f.path for f in os.scandir(settings.set.absolute_test_folder) if f.is_dir()])
            if len(subfolders) == 0:
                logging.warning("WARNING - no test folders found in: " + settings.set.absolute_test_folder)
            else:
                for folder in subfolders:
                    head_tail = os.path.split(folder)
                    testname = head_tail[1]
                    subsubfolders = sorted([f.path for f in os.scandir(folder) if f.is_dir()])
                    if len(subsubfolders) == 0:
                        logging.warning("WARNING - no scene folders found for test in " + folder)
                        continue
                    for folder2 in subsubfolders:
                        head_tail = os.path.split(folder2)

                        scenefound = False
                        for scene in self.scenes:
                            if scene['folder_name'] == head_tail[1]:
                                scene['tests'][testname] = {}
                                scene['tests'][testname]['average_test_time'] = 0.0
                                scene['tests'][testname]['test_path'] = folder2
                                scene['tests'][testname]['lr_path'] = scene['lr_path']
                                scene['tests'][testname]['hr_path'] = scene['hr_path']
                                scenefound = True
                                break
                        if not scenefound:
                            logging.warning("WARNING - no matching scene found for " + folder2 + " scenename="+head_tail[1])

        logging.info("INFO - saving database")
        self.save()

    def make_scene_sheet(self):
        import cv2
        LARGE_FONT_SIZE = 20
        SMALL_FONT_SIZE = 8

        SCENE_SHEET_FRAME_WIDTH = 100
        SCENE_SHEET_FRAME_MARGIN = 2
        SCENE_SHEET_PADDING = 10
        SCENE_SHEET_FRAMES_PER_ROW = 10

        # assume 4/3 aspect ratio
        frame_width = SCENE_SHEET_FRAME_WIDTH
        frame_height = int(100 * 3 / 4)

        title = "Scene Sheet"
        margin_left = SCENE_SHEET_FRAME_MARGIN
        margin_top = SCENE_SHEET_FRAME_MARGIN
        margin_right = SCENE_SHEET_FRAME_MARGIN
        margin_bottom = SCENE_SHEET_FRAME_MARGIN

        NUM_DESCRIPTION_LINES = 3

        # get the images
        imgs = []
        descriptions1 = []
        descriptions2 = []
        descriptions3 = []
        for scene in self.scenes:
            # get description strings for the scene
            descriptions1.append(str(scene['scene_index']) + "-" + scene['folder_name'])
            descriptions2.append(scene['video_name'])
            descriptions3.append("("+ str(scene['start_frame']) + "-" + str(scene['end_frame']) +") "+ str(scene['num_frames'])+" frames")

            # see if there is a high resolution/ground truth image
            if 'hr_path' in scene and scene['hr_path'] != '':
                if os.path.exists(scene['hr_path']):
                    hrfiles = utils.GetFiles(os.path.join(scene['hr_path']))
                    if os.path.exists(hrfiles[0]):
                        img = Image.open(hrfiles[0]).resize((frame_width, frame_height))
                        imgs.append(img)
                        continue

            # otherwise try getting the frame from the video
            if 'video_name' in scene and scene['video_name'] != '':
                videopath = os.path.join(settings.set.absolute_videos_folder, scene['video_name'])
                video = cv2.VideoCapture(videopath)
                if not os.path.exists(videopath):
                    logging.warning("WARNING - video can not be found " + videopath + "for scene: " + scene['video_name'] )
                else:
                    video.set(1, scene['start_frame'])
                    ret, cv2_im =  video.read()
                    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_im)
                    img = img.resize((frame_width, frame_height))
                    imgs.append(img)
                    continue

            # otherwise just use a black frame
            img = Image.new('RGB', (frame_width, frame_height))
            imgs.append(img)

        # rows and columns for the sheet
        ncols = SCENE_SHEET_FRAMES_PER_ROW
        nrows = math.ceil(len(self.scenes)/ncols)

        padding = SCENE_SHEET_PADDING

        # title appears at the top of image
        TITLEHEIGHT = 30

        # descriptions appear below the photos
        DESCRIPTION_HEIGHT = 12

        WHITE_COLOR = (255, 255, 255)

        # compute various dimensions (margins, etc)
        marw = margin_left + margin_right
        marh = margin_top + margin_bottom
        padw = (ncols - 1) * padding

        # determine overall width/height of the image
        imagewidth = ncols * frame_width + marw + padw
        imageheight = TITLEHEIGHT + marh + nrows * (
                    frame_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
        isize = (imagewidth, imageheight)

        # Create the new image. The background doesn't have to be white
        inew = Image.new('RGB', isize, WHITE_COLOR)

        # create the fonts.  Will use FreeSans.ttf in fonts-freefont-ttf
        font_path = os.path.join("FAIME","resources","FreeSans.ttf")
        font = ImageFont.truetype(font_path, LARGE_FONT_SIZE)
        fontsmall = ImageFont.truetype(font_path, SMALL_FONT_SIZE)

        # draw lines and text for each frame
        numframes = len(imgs)

        # create a draw object to draw text and lines on
        draw = ImageDraw.Draw(inew)

        # draw title
        sizew, sizeh = draw.textsize(title, font)
        draw.text((int(imagewidth / 2 - sizew / 2), int(TITLEHEIGHT / 2 - sizeh / 2)), title, (0, 0, 0), font=font,
                  align="ms")

        # draw lines and text for each frame
        frameindex = 0

        # For each row in the grid
        for irow in range(nrows):
            # for each column in the grid
            for icol in range(ncols):
                # does this grid cell contain an image
                if frameindex < numframes:
                    # calculate the bounding box for the image in the grid cell
                    left = margin_left + icol * (frame_width + padding)
                    right = left + frame_width
                    upper = TITLEHEIGHT + margin_top + irow * (
                                frame_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
                    lower = upper + frame_height
                    bbox = (left, upper, right, lower)

                    # get the image to put in the grid cell
                    try:
                        img = imgs.pop(0)
                    except:
                        break
                    inew.paste(img, bbox)

                    # compute dimensions for the image border
                    left = margin_left + icol * (frame_width + padding)
                    right = left + frame_width
                    upper = TITLEHEIGHT + margin_top + irow * (
                                frame_height + padding + NUM_DESCRIPTION_LINES * DESCRIPTION_HEIGHT)
                    lower = upper + frame_height
                    bbox = (left, upper, right, lower)

                    # draw a colored rectangle around the frame
                    draw.rectangle(bbox, outline=(0,0,0), fill=None, width=2)

                    # Description line 1- draw the frame number
                    center = int((left + right) / 2)
                    middle = int(lower + DESCRIPTION_HEIGHT / 2)
                    sizew, sizeh = draw.textsize(descriptions1[frameindex], fontsmall)
                    draw.text((int(center - sizew / 2), int(middle - sizeh / 2)), descriptions1[frameindex], (0, 0, 0), font=fontsmall,
                              align="ms")
                    # Description line 2 - draw the video name
                    middle = int(lower + DESCRIPTION_HEIGHT * 3 / 2)
                    sizew, sizeh = draw.textsize(descriptions2[frameindex], fontsmall)
                    draw.text((int(center - sizew / 2), int(middle - sizeh / 2)), descriptions2[frameindex], (0, 0, 0), font=fontsmall,
                              align="ms")

                    # Description line 3 - frame start-end
                    middle = int(lower + DESCRIPTION_HEIGHT * 5 / 2)
                    sizew, sizeh = draw.textsize(descriptions3[frameindex], fontsmall)
                    draw.text((int(center - sizew / 2), int(middle - sizeh / 2)), descriptions3[frameindex], (0, 0, 0), font=fontsmall,
                              align="ms")
                # on to next grid cell
                frameindex = frameindex + 1

        # save the scene sheet
        filename = os.path.join(settings.set.absolute_project_folder, "scene_sheet.png")
        inew.save(filename)

    def save(self):
        #logging.info("Saving database")
        self.database_filename = os.path.join(os.path.abspath(settings.set.project_folder), "database.json")
        with open(self.database_filename, 'w') as f:
            json.dump(self.scenes, f, indent=2, separators=(',', ':'))

    def clear(self):
        #logging.info("Clearing database")
        self.scenes.clear()

        # remove LR folder
        if os.path.exists(settings.set.absolute_LR_folder):
            utils.RemoveDir(settings.set.absolute_LR_folder)

        # remove CHR folder
        if os.path.exists(settings.set.absolute_CHR_folder):
            utils.RemoveDir(settings.set.absolute_CHR_folder)

        # remove Test folder
        if os.path.exists(settings.set.absolute_test_folder):
            utils.RemoveDir(settings.set.absolute_test_folder)

        # remove vqt folder
        if os.path.exists(settings.set.absolute_vqt_folder):
            utils.RemoveDir(settings.set.absolute_vqt_folder)

        # remove the database
        self.database_filename = os.path.join(os.path.abspath(settings.set.project_folder), "database.json")
        if os.path.exists(self.database_filename):
            os.remove(self.database_filename)

        # remove the log
        logfilename = os.path.join(os.path.abspath(settings.set.project_folder), "log.csv")
        if os.path.exists(logfilename):
            os.remove(logfilename)

        # remove the scenesheet
        scenesheetfilename = os.path.join(settings.set.absolute_project_folder, "scene_sheet.png")
        if os.path.exists(scenesheetfilename):
            os.remove(scenesheetfilename)

    def add(self, scenetoadd, createfoldername):

        maxindex = -1
        for scene in self.scenes:
            if scene['scene_index'] > maxindex:
                maxindex = scene['scene_index']
        scenetoadd['scene_index'] = maxindex+1
        if createfoldername:
            scenetoadd['folder_name'] = "scene" + "_%04d" % (maxindex+1)
        self.scenes.append(scenetoadd)
        pass

    def getNumScenes(self):
        return len(self.scenes)

    def getScene(self, index):
        return self.scenes[index]

    def containsVideo(self, name):
        for scene in self.scenes:
            if scene['video_name'] == name:
                return True
        return False

    def getVideos(self):
        videos = []
        for scene in self.scenes:
            if not scene['video_name'] in videos:
                videos.append(scene['video_name'])
        return videos

    def getScenesForVideo(self, video):
        vscenes = []
        for scene in self.scenes:
            if scene['video_name'] == video:
                vscenes.append(scene)
        return vscenes

    def list(self):
        if len(self.scenes) == 0:
            print("No scenes in database")
            return

        logging.info("Scenes in the database:")
        headers = ["scene index",
                   "folder",
                   "video",
                   "start",
                   "end",
                   "# frames",
                   "width",
                   "height",
                   "bit depth",
                   "cropped",
                   "cropped # frames",
                   "cropped width",
                   "cropped height",
                   "fps",
                   "duplicated frames",
                   "same size"]
        table = []
        for scene in self.scenes:
            table.append([scene.get('scene_index','Not Found'),
                          scene.get('folder_name','Not Found'),
                          scene.get('video_name','Not Found'),
                          scene.get('start_frame','Not Found'),
                          scene.get('end_frame','Not Found'),
                          scene.get('num_frames','Not Found'),
                          scene.get('frame_width','Not Found'),
                          scene.get('frame_height','Not Found'),
                          scene.get('bit_depth', 'Not Found'),
                          scene.get("cropped",'Not Found'),
                          scene.get("cropped_num_frames",'Not Found'),
                          scene.get("cropped_frame_width",'Not Found'),
                          scene.get("cropped_frame_height",'Not Found'),
                          scene.get('fps','Not Found'),
                          scene.get('duplicated_frames','Not Found'),
                          scene.get('same_size','Not Found')])

        report = tabulate(table, headers, tablefmt="plain")
        for item in report.split('\n'):
            logging.info(item)

    def getSceneIndices(self):
        if len(settings.set.individual_scenes) > 0:
            return settings.set.individual_scenes
        else:
            return list(range(len(self.scenes)))

    def findSceneFromFolderName(self, foldername):
        for scene in self.scenes:
            if scene['folder_name'] == foldername:
                return scene
        return None

    def removescenes(self, scenestoremove):
        for sceneid in scenestoremove:
            scenedeleted = False
            for index, scene in enumerate(self.scenes):
                if scene['scene_index'] == sceneid:
                    logging.info("Removing scene from database: "+ str(sceneid))

                    # Delete HR folder
                    if 'hr_path' in scene.keys() and scene['hr_path'] != '':
                        if os.path.exists(scene['hr_path']):
                            shutil.rmtree(scene['hr_path'])

                    # Delete LR folder
                    if 'lr_path' in scene.keys() and scene['lr_path'] != '':
                        if os.path.exists(scene['lr_path']):
                            shutil.rmtree(scene['lr_path'])

                    # Delete CHR folder
                    if 'chr_path' in scene.keys() and scene['chr_path'] != '':
                        if os.path.exists(scene['chr_path']):
                            shutil.rmtree(scene['chr_path'])

                    # Delete test folder
                    if 'test_path' in scene.keys() and scene['test_path'] != '':
                        if os.path.exists(scene['test_path']):
                            shutil.rmtree(scene['test_path'])

                    del self.scenes[index]
                    scenedeleted = True
                    break
            if not scenedeleted:
                logging.warning("Unable to find scene in database: " + str(sceneid))
        self.save()
        self.list()

db = Database()