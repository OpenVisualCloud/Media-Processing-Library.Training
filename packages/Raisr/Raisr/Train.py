# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import cv2
import numpy as np
import os
import pickle
import sys
from Raisr import Utils
from math import floor
import logging
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

R = None
PATCH_SIZE = None
GRADIENT_SZIE = None
Q_ANGLE = None
Q_STRENGTH = None
Q_COHERENCE = None
STREBIN = None
COHEBIN = None
MARGIN = None
PATCH_MARGIN = None
GRADIENT_MARGIN = None
Q = None
V = None
H = None
WEIGHTING = None

INSTANCE = None
DONE_PATCHES = None
QUANTIZATION = None
args = None

def save_filters():
    save_path = args.filterPath
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    global H
    H = H.reshape((Q_ANGLE*Q_STRENGTH*Q_COHERENCE, R * R, PATCH_SIZE * PATCH_SIZE))  
    Utils.save3DASCII(H, os.path.join(save_path,"filterbin_"+str(R)+"_"+str(args.bitdepth)))
    Utils.saveASCII_F(STREBIN, os.path.join(save_path,"Qfactor_strbin_"+str(R)+"_"+str(args.bitdepth)))
    Utils.saveASCII_F(COHEBIN, os.path.join(save_path,"Qfactor_cohbin_"+str(R)+"_"+str(args.bitdepth)))

    with open(os.path.join(save_path,"filter_"+str(R)+"_"+str(args.bitdepth)), "wb") as fp:
        pickle.dump(H, fp)

    with open(os.path.join(save_path,"Qfactor_str_"+str(R)+"_"+str(args.bitdepth)), "wb") as sp:
        pickle.dump(STREBIN, sp)

    with open(os.path.join(save_path,"Qfactor_coh_"+str(R)+"_"+str(args.bitdepth)), "wb") as cp:
        pickle.dump(COHEBIN, cp)

def gettrainargs(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-ohf", "--hr_path")
    parser.add_argument("-chf", "--chr_path")
    parser.add_argument("-lf", "--lr_path")
    parser.add_argument("-ff", "--filterPath")
    parser.add_argument("-bits", "--bitdepth")
    args = parser.parse_args(args = args)
    return args
def main(passed_args=None,params=None):
    from argparse import Namespace
    global args
    global R
    global PATCH_SIZE
    global GRADIENT_SZIE
    global Q_ANGLE
    global Q_STRENGTH
    global Q_COHERENCE
    global STREBIN
    global COHEBIN
    global MARGIN
    global PATCH_MARGIN
    global GRADIENT_MARGIN
    global Q
    global V
    global H
    global WEIGHTING
    global INSTANCE
    global DONE_PATCHES
    global QUANTIZATION

    parsed_args = gettrainargs(params)
    args = Namespace(**vars(parsed_args),**passed_args)
    # Define parameters
    R = args.scale_factor#up scale
    PATCH_SIZE = args.patch_size
    GRADIENT_SIZE = args.gradient_size
    Q_ANGLE = args.angle_quantization
    Q_STRENGTH = args.strength_quantization
    Q_COHERENCE = args.coherence_quantization
    INSTANCE = 30000000                          # use 20000000 patches to get the Strength and COHErence boundary
    DONE_PATCHES = 0                             # patch number has been used
    QUANTIZATION = np.zeros((INSTANCE,2))        # QUANTIZATION boundary
    STREBIN = np.zeros((Q_STRENGTH-1))  # Strength boundary
    COHEBIN = np.zeros((Q_COHERENCE-1)) # Coherence boundary
    # Calculate the margin
    maxblocksize = max(PATCH_SIZE, GRADIENT_SIZE)
    MARGIN = floor(maxblocksize/2)
    PATCH_MARGIN = floor(PATCH_SIZE/2)
    GRADIENT_MARGIN = floor(GRADIENT_SIZE/2)
    Q = np.zeros((Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R*R, PATCH_SIZE*PATCH_SIZE, PATCH_SIZE*PATCH_SIZE))
    V = np.zeros((Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R*R, PATCH_SIZE*PATCH_SIZE))
    H = np.zeros((Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R*R, PATCH_SIZE*PATCH_SIZE))

    # Matrix preprocessing
    # Preprocessing normalized Gaussian matrix W for hashkey calculation
    WEIGHTING = Utils.gaussian2d([GRADIENT_SIZE, GRADIENT_SIZE], 2)
    WEIGHTING = WEIGHTING/WEIGHTING.max()
    WEIGHTING = np.diag(WEIGHTING.ravel())

    hr_imagelist= args.hr_images
    chr_imagelist= None if not args.chr_images else args.chr_images
    lr_imagelist= None if not args.lr_images else args.lr_images

    if not args.chr_path and not args.lr_path:
        logging.info('Raisr..Running original Raisr Algo (original HR -> LR -> cheap-upscale)')
    elif args.chr_path  and not args.lr_path:
        logging.info('Raisr..Running RAISR++ algo (load cheap-upscaled HR image)')
    elif args.lr_path and not args.chr_path:
        logging.info("Raisr..Running original RAISR algo (load LR -> cheap-upscale)")
    else:
        logging.warning("Raisr..Invalid parameters")
        return
    start = time.time()
    logging.info('Raisr..TrainingImages: {}'.format(np.size(hr_imagelist)))
    per_img_patches = int(INSTANCE/np.size(hr_imagelist)/2 + 1)
    logging.info('Raisr..PatchesPerImg: {}'.format(per_img_patches))

    for index,image in enumerate(hr_imagelist):
        print('\r', end='')
        print('' * 60, end='')
        print('\r Quantization: Processing '+ str(INSTANCE/2) + ' patches (' + str(200*DONE_PATCHES/INSTANCE) + '%)')
        # Extract only the luminance in YCbCr
        grayorigin = Utils.loadYImg(image, args.bitdepth,R)
        if not args.chr_path and not args.lr_path:
            upscaledLR = Utils.GetUpscaleLRFromHR(grayorigin,R,args.bitdepth)
        elif args.chr_path and not args.lr_path:
            upscaledLR = Utils.GetUpscaleLRFromCHR(index,chr_imagelist,args.bitdepth,R)
        elif args.lr_path and not args.chr_path:
            upscaledLR = Utils.GetUpscaleLRFromLR(index,lr_imagelist,args.bitdepth,R)
        # Normalized to [0,1]
        grayorigin = Utils.normalize(grayorigin,args.bitdepth)
        im_GX,im_GY = np.gradient(upscaledLR)
        try:
            QUANTIZATION,DONE_PATCHES = Utils.quantization(im_GX,im_GY,DONE_PATCHES,per_img_patches,R,WEIGHTING,QUANTIZATION,MARGIN,GRADIENT_MARGIN)
        except Exception as e:
            logging.error('Raisr..Error in Quantization')
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)
            raise e
        if DONE_PATCHES>INSTANCE/2:
            break
    QUANTIZATION = np.sort(QUANTIZATION[0:DONE_PATCHES,:],axis=0)
    for i in range (1,Q_STRENGTH):
        STREBIN[i-1] = QUANTIZATION[floor(i*DONE_PATCHES/Q_STRENGTH),0]
    for i in range (Q_COHERENCE):
        COHEBIN[i-1] = QUANTIZATION[floor(i*DONE_PATCHES/Q_STRENGTH),1]        

    imagecount = 1
    logging.info('Raisr..Begin to process images: ')
    for index,image in enumerate(hr_imagelist):
        print('\r', end='')
        print(' ' * 60, end='')
        print('\r Processing image ' + str(imagecount) + ' of ' + str(len(hr_imagelist)) + ' (' + image + ')')
        grayorigin = Utils.loadYImg(image, args.bitdepth,R)
        if not args.chr_path and not args.lr_path:
            upscaledLR = Utils.GetUpscaleLRFromHR(grayorigin,R,args.bitdepth)
        elif args.chr_path and not args.lr_path:
            print("cur chr:%s"%chr_imagelist[index])
            upscaledLR = Utils.GetUpscaleLRFromCHR(index,chr_imagelist,args.bitdepth,R)
        elif args.lr_path and not args.chr_path:
            print("cur lr:%s"%lr_imagelist[index])
            upscaledLR = Utils.GetUpscaleLRFromLR(index,lr_imagelist,args.bitdepth,R)
        # Normalized to [0,1]
        grayorigin = Utils.normalize(grayorigin,args.bitdepth)
        height, width = upscaledLR.shape
        im_GX,im_GY = np.gradient(upscaledLR)
        theta = np.zeros((upscaledLR.shape))
        lamda = np.zeros((upscaledLR.shape))
        u = np.zeros((upscaledLR.shape))
        try:
            Q,V = Utils.cal_qv_split(Q,V,MARGIN,height,width,upscaledLR,grayorigin,im_GX,im_GY,STREBIN,COHEBIN,PATCH_MARGIN,GRADIENT_MARGIN,Q_ANGLE,WEIGHTING,theta, lamda, u,R)
        except Exception as e:
            logging.error('Raisr..Error in Train Process')
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)
            raise e
        imagecount += 1

    # Preprocessing permutation matrices P for nearly-free 8x more learning examples
    print('\r', end='')
    print(' ' * 60, end='')
    print('\r Preprocessing permutation matrices P for nearly-free 8x more learning examples ...')
    sys.stdout.flush()
    P=Utils.cal_p(PATCH_SIZE)

    Qextended = np.zeros((Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R*R, PATCH_SIZE*PATCH_SIZE, PATCH_SIZE*PATCH_SIZE))
    Vextended = np.zeros((Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R*R, PATCH_SIZE*PATCH_SIZE))
    Qextended,Vextended = Utils.cal_qextend(R,Q_ANGLE,Q_STRENGTH,Q_COHERENCE,P,Q,V,Qextended,Vextended)
    Q += Qextended
    V += Vextended

    # Compute filter h
    print('\r', end='')
    print(' ' * 60, end='')
    print('\r Computing H ...')
    sys.stdout.flush()
    H = Utils.cal_h(R,Q_ANGLE,Q_STRENGTH,Q_COHERENCE,H,Q,V)

    # Write filter to file
    save_filters()
    end = time.time()
    total_secs = end - start
    if total_secs < 60:
        logging.info('Raisr..Time used in seconds: {}'.format(total_secs))
    else:
        logging.info('Raisr..Time used in minutes: {}'.format(floor(total_secs/60)))
    print('\r', end='')
    print(' ' * 60, end='')
    logging.info('Raisr..Finished.')
