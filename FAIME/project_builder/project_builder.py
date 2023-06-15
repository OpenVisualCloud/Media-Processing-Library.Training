# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from FAIME.video import Video

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.y4m', '.ppm', '.PPM', '.bmp', '.BMP',]

VID_EXTENSIONS = ['.wemb','.mkv','.gif','.wmv',
    '.mov','.m4v','.mp4','.m4p','.mpg']

def parse_args(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--media_folder',type=str,help='Path to media storage folder')
    parser.add_argument('--project_folder',type=str,help='Path to project folder to create')
    parser.add_argument('--img_per_vid',type=int,default=-1,help='How many images to pull per video')
    parser.add_argument('--percent_frames',type=float,default=.1,help='When img_per_vid not provided, select a percentage of frames')
    parser.add_argument('--reset_project',type=bool, default=False, help='Delete all folders when true')
    parser.add_argument('--verbose',type=bool,default=False,help='Print frames selected')
    parser.add_argument('--add_images',type=bool,default=True,help='Add images to HR folder of project')
    return parser.parse_args(args=args)

""" Sort images by characteristics
    1) Resolution
    2) Bit Depth
"""

def main(args=None):
    import cv2
    import ffmpeg
    import logging
    import math
    import os
    import random
    import shutil
    import sys
    from pathlib import Path


    logger = logging.getLogger()

    # Setup Args and Paths
    args = parse_args(args=args)
    load_path = args.media_folder
    save_path = os.path.join(args.project_folder,'HR')
    os.makedirs(save_path,exist_ok=True)

    # Reset project folder if requested
    if args.reset_project:
        for item in os.listdir(args.project_folder):
            path = os.path.join(args.project_folder,item)
            if os.path.isdir(path):
                shutil.rmtree(path,ignore_errors=True)
            else:
                os.remove(path)
    elif not args.add_images:
        import re
        # Use a regular expression to check for image and video file extensions
        image_regex = re.compile(r'\.(jpg|jpeg|png|gif|tiff|bmp)$')
        video_regex = re.compile(r'\.(mp4|avi|mov|mkv|webm)$')
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            if image_regex.search(file_path):
                logging.warning('ProjectBuilder..WARNING: Image files already exist in project folder but --project_builder_reset_project is set to False')
                return
            elif video_regex.search(file_path):
                logging.warning('ProjectBuilder..WARNING: Video files already exist in project_folder but --project_builder_reset_project is set to False')
                return

    # Prop is a tuple of resolution and bit depth. Matching props are stored in matching folders
    props = set()
    prop_mappings = {}
    max_folder = 0

    if not os.path.exists(load_path):
        logging.warning('ProjectBuilder..WARNING: --media_folder does not exist')
        return
    elif not os.path.isdir(load_path):
        logging.warning('ProjectBuilder..WARNING: --media_path is not a directory')
        return

    img_nums = [] # A list of image counts for each directory.

    for root, _, files in os.walk(load_path):
        for file in files:
            # Grab extension and create full path
            ext = os.path.splitext(file)[1]
            full_path = os.path.join(root, file)
            # folder_num: format = scene_XXXX
            # max_folder, updated value for the next folder to be created
            # num, the XXXX part of folder_num
            folder_num, max_folder, num= GetFolderNum(prop_mappings, max_folder, full_path)
            # Always attempt to create folder even if it exists
            Path(os.path.join(save_path, folder_num)).mkdir(parents=True,exist_ok=True)
            # A check to see if we need a new image index
            if len(img_nums) <= num:
                img_nums.append(0)
            img_num = img_nums[num]

            # File is an image
            if ext in IMG_EXTENSIONS:
                # Copy the full path over to save_path/folder_num/XXXX.ext
                shutil.copyfile(full_path,os.path.join(save_path,folder_num,'{:0>4d}{}'.format(img_num,ext)))
                img_nums[num] += 1
            elif ext in VID_EXTENSIONS:
                cap = Video(full_path,'%04d'%num)
                # If img_per_vid not provided, use percent of frames
                num_images = args.img_per_vid
                if num_images == -1:
                    num_images = math.floor(cap.frames * args.percent_frames)
                # Get a set of random frames to extract
                img_indexes = set()
                while len(img_indexes) < num_images:
                    img_indexes.add(random.randint(-1,cap.frames))

                # With 8bit videos we can easily use cv2
                if not cap.is10bit:
                    cap.open()
                    for i in img_indexes:
                        img_num = img_nums[num]
                        succ = cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
                        ret, img = cap.read()
                        cv2.imwrite(os.path.join(save_path,folder_num,'%04d.png'%img_num),img)
                        img_nums[num] += 1
                # But with 1-bit videos we need to use FFmpeg
                else:
                    frame_string = '+'.join(['eq(n,{})'.format(i) for i in img_indexes])
                    logging.info('ProjectBuilder..{} contributing frames [{}] to folder {}'.format(file,','.join([str(i) for i in img_indexes]),folder_num))
                    (ffmpeg
                    .input(full_path)
                    .filter('select',frame_string)
                    .output(os.path.join(save_path,folder_num,"%04d.png"),**{'qscale:v':3,'pix_fmt':'rgb48le','vsync':0,'start_number':img_num})
                    .run(quiet=True))
                    img_nums[num] += num_images
            # Write the file and a tabbed series of frames used
            (open(os.path.join(save_path,folder_num,'metadata.txt'),'a')
                 .write('file {} : {} bit {}\n'.format(file, Video.getBits(full_path),Video.getResolution(full_path))))
            (open(os.path.join(save_path,folder_num,'metadata.txt'),'a')
                .write('\tframes: {}'.format(','.join(map(str,img_indexes)))))

def GetFolderNum(prop_mappings, max_folder, file):
    if Video.is10bitStatic(file):
        if (Video.getResolution,'10') in prop_mappings:
            folder_num = prop_mappings[(Video.getResolution(file),'10')]
        else:
            folder_num = max_folder
            max_folder += 1
            prop_mappings[(Video.getResolution(file),'10')] = folder_num
    else:
        if (Video.getResolution,'8') in prop_mappings:
            folder_num = prop_mappings[(Video.getResolution(file),'8')]
        else:
            folder_num = max_folder
            max_folder += 1
            prop_mappings[(Video.getResolution(file),'8')] = folder_num
    return ('scene_%04d' % folder_num, max_folder, folder_num)

if __name__ == '__main__':
    main()