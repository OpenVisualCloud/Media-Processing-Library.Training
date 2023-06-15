# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numba as nb
import os
import cv2
from argparse import ArgumentParser
import warnings
from inspect import _void
from Raisr.Y4M import *
from pathlib import Path
import logging
warnings.filterwarnings("ignore")


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("-bits", dest="bitdepth", default='10',
                        help="bit depth of input image")
    parser.add_argument("-i", dest="InputPath", default='4k10bit_train_noNBC',
                        help="Original image folder")
    parser.add_argument("-o", dest="OutputPath", default='4k10bit_train_noNBC_sharpen_1',
                        help="sharpened image folder")
    parser.add_argument("-s", dest="sharpen_amount", default="1.0",
                        help="sharpen amount parameter")
    args = parser.parse_args(args=args)
    return args

def main(passed_args=None, params=None):
    from argparse import Namespace
    parsed_args = parse_args(params)
    args = Namespace(**vars(parsed_args),**passed_args)
    if not args.hr_images:
        logging.warning('Raisr.Sharpen..No input path provided. Exiting...')
        return
    Infilelist = args.hr_images
    logging.warning("Raisr.Sharpen..Input images: {}".format(np.size(Infilelist)))
    for image in Infilelist:
        print("sharpening: ", image)
        parser = Reader(_void,verbose=False)
        with open(image, 'rb') as fh:
            data = fh.read()
            if not data:
                return
            y4mtuple = parser.decode(data)
            im_1d = np.frombuffer(y4mtuple[0],dtype=np.uint16)
            im = np.reshape(im_1d, (-1,y4mtuple[2]['W']))
        sharped_image = unsharp_mask(image = im,bits = 10,amount=float(args.sharpen_amount))
        # Get the base path of the new file in the sharpen_path
        newfile_basepath = os.path.join(
                os.path.basename(os.path.dirname(image)),
                Path(image).stem + '.y4m')
        # Create the folder for the new file in the sharpen_path
        Path(os.path.join(args.OutputPath,os.path.dirname(newfile_basepath))
                ).mkdir(parents=True,exist_ok=True)
        filepath = os.path.join(args.OutputPath,newfile_basepath)
        generator = Writer(open(filepath,'wb'))
        generator.encode(Frame(sharped_image.tobytes(),y4mtuple[1],y4mtuple[2],1,0))

def unsharp_mask(image, bits, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    if (bits == 8):
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    if (bits == 10):
        sharpened = np.minimum(sharpened, 1023 * np.ones(sharpened.shape))
    if (bits == 16):
        sharpened = np.minimum(sharpened, 65535 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint16)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened









