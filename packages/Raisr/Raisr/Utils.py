# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from math import atan2, floor, pi, isnan
import numba as nb
import cv2
import os
from math import floor
from scipy import interpolate
from skimage import transform
import glob
from numba import prange
from Y4M import *
from inspect import _void
def gaussian2d(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def GetFiles(directory: str, recursive: bool = False):
    """
    This function returns a list of all files with the specified extensions in the given directory.
    If the recursive flag is set to True, it will also include files in subdirectories of the given directory.
    The file extensions to include in the list are defined in the 'types' tuple.
    The list of files is sorted in alphabetical order.
    """
    # Define the file types to include in the list
    types = ('*.png', '*.jpg', '*.y4m')

    # Initialize an empty list to hold the files
    inputfiles = []

    # Loop over the file types
    for files in types:
        # Add all files with the current type to the list
        inputfiles.extend(glob.glob(os.path.join(directory, files)))

    # If the recursive flag is True, get files in subdirectories
    if recursive:
        # Loop over the subdirectories in the current directory
        for subdir in os.listdir(directory):
            # Get the full path to the subdirectory
            d = os.path.join(directory, subdir)
            # If the path is a directory, get its files
            if os.path.isdir(d):
                # Add the files in the subdirectory to the list
                inputfiles.extend(GetFiles(d, recursive))
    # Sort the files in alphabetical order
    inputfiles.sort()
    # Return the list of files
    return inputfiles
def isY(image):
    x = abs(image[:,:,0]-image[:,:,1])
    y = np.linalg.norm(x)
    y_flag = True if y==0 else False
    return y_flag
    
def img2Y(img_path):
    im_uint8 = cv2.imread(img_path)
    if (im_uint8 is None):
        print("load image fails! exit! If you are using -chf or -lf, filename needs to be same as -ohf")
        exit()
    if isY(im_uint8):
        im_uint8 = im_uint8[:,:,0]     
    if len(im_uint8.shape)>2:
        y_img = cv2.cvtColor(im_uint8, cv2.COLOR_BGR2YCrCb)[:,:,0]
    else:
        y_img = im_uint8
    return y_img


def rCrop(image,R):
    shape = image.shape
    h = shape[0]
    w = shape[1]
    hnew = h - h % R
    wnew = w - w % R
    if len(image.shape) == 2:
        crop_res = image[0:hnew, 0:wnew]
    else:
        crop_res = image[0:hnew, 0:wnew, :]
    return crop_res

def loadYImg(image, bits,R):
    if bits == '10':
        parser = Reader(_void, verbose=False)
        with open(image, 'rb') as fh:
            data = fh.read()
            if not data:
                return
            y4mtuple = parser.decode(data)
        im_1d = np.frombuffer(y4mtuple[0], dtype=np.uint16)
        im = np.reshape(im_1d, (-1, int(y4mtuple[2]['W'])))
    else:
        # load HR image
        im = img2Y(image)
    im = rCrop(im, R)
    return im    

# downscale -> upscale
def PrepareCV(im, R):
    h, w = im.shape
    size_tmp = tuple((np.array((w,h)) / R).astype(int))

    imL = cv2.resize(im,size_tmp,interpolation=cv2.INTER_LINEAR)
    imCheapUpscaled = cv2.resize(imL,(w,h),interpolation=cv2.INTER_LINEAR)
    return imCheapUpscaled

# upscale only
def CheapUpscale(im, R):
    h, w = im.shape
    imCheapUpscaled = cv2.resize(im,(w*R,h*R),interpolation=cv2.INTER_LINEAR)
    return imCheapUpscaled

def normalize(image, bits):
    if image.dtype == 'uint8':
        res = cv2.normalize(image.astype('float'), None, image.min()/255, image.max()/255, cv2.NORM_MINMAX)
    elif image.dtype == 'uint16' and bits == '10':
        res = cv2.normalize(image.astype('float'), None, image.min()/1023, image.max()/1023, cv2.NORM_MINMAX)
    elif image.dtype == 'uint16' and bits == '16':
        res = cv2.normalize(image.astype('float'), None, image.min()/65535, image.max()/65535, cv2.NORM_MINMAX)
    elif image.dtype == 'float':
        res = image
    res = np.clip(res, 0, 1)
    return res

def GetUpscaleLRFromHR(grayorigin,R,bits):
    grayorigin_hrtochr = PrepareCV(grayorigin,R)
    upscaledLR = normalize(grayorigin_hrtochr,bits)
    return upscaledLR

def GetUpscaleLRFromCHR(index,chr_imagelist,bits,R):
        grayorigin_chr = loadYImg(chr_imagelist[index], bits,R)
        upscaledLR = normalize(grayorigin_chr,bits)
        return upscaledLR

def GetUpscaleLRFromLR(index,lr_imagelist,bits,R):
        grayorigin_lr = loadYImg(lr_imagelist[index], bits,R)
        upLR = CheapUpscale(grayorigin_lr, R)
        upscaledLR = normalize(upLR,bits)
        return upscaledLR

@nb.jit(nopython=True, parallel=True) 
def cgls(A, b):
    height, width = A.shape
    x = np.zeros((height))
    while(True):
        sumA = A.sum()
        if (sumA < 100):
            break
        if (np.linalg.det(A) < 1):
            A = A + np.eye(height, width) * sumA * 0.000000005
        else:
            x = np.linalg.inv(A).dot(b)
            break
    return x

@nb.jit(nopython=True, parallel=True)
def quantization(im_GX,im_GY,done_patches,per_img_patches,R,weighting,quantiMatrix,margin,gradientmargin):
    H,W = im_GX.shape
    rest_patches = per_img_patches
    while rest_patches>=1:
        i1 = np.random.randint(margin,H-margin-R+1)
        j1 = np.random.randint(margin,W-margin-R+1)   
        for i2 in range(R):
            for j2 in range(R):
                i = i1+i2
                j = j1+j2
                gradientblockX = im_GX[i-gradientmargin:i+gradientmargin+1, j-gradientmargin:j+gradientmargin+1]
                gradientblockY = im_GY[i-gradientmargin:i+gradientmargin+1, j-gradientmargin:j+gradientmargin+1]
                strength, coherence = GetStrCoh(gradientblockX,gradientblockY,weighting)
                quantiMatrix[done_patches,0] = strength
                quantiMatrix[done_patches,1] = coherence
                done_patches+=1
                rest_patches-=1
    return quantiMatrix,done_patches

@nb.jit(nopython=True, parallel=True)
def cal_qv_split(Q,V,margin,height,width,upscaledLR,grayorigin,im_GX,im_GY,strbin,cohbin,patchmargin,gradientmargin,Qangle,weighting,theta, lamda, u,R):
    for row in prange(margin, height-margin):
        for col in range(margin, width-margin):
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patchL = np.expand_dims(patch.ravel(), axis = 0)
            # Get gradient block
            gradientblockX = im_GX[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            gradientblockY = im_GY[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkeyJIT(gradientblockX,gradientblockY, Qangle, weighting,strbin,cohbin)
            theta[row][col] = angle
            lamda[row][col] = strength
            u[row][col] = coherence
    for k1 in prange(24):
        for row in range(margin, height-margin):
            for col in range(margin, width-margin):
                angle = theta[row][col]
                strength = lamda[row][col]
                coherence = u[row][col]
                angle = np.int(angle)
                strength = np.int(strength)
                coherence = np.int(coherence)
                if(angle%24 == k1):#
                    patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
                    patchL = np.expand_dims(patch.ravel(), axis = 0)
                    pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
                    pixelHR = grayorigin[row,col]
                    ATA = np.dot(patchL.T, patchL)
                    p1 = patchL.T
                    ATb = p1.reshape((p1.size))
                    for k2 in range(p1.size):
                        ATb[k2] = ATb[k2] * pixelHR
                    ATb = ATb.ravel()
                    Q[angle,strength,coherence,pixeltype] += ATA
                    V[angle,strength,coherence,pixeltype] += ATb
    return Q,V

#@nb.jit(nopython=True, parallel=True)
def cal_p(patchsize):
    P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7))
    rotate = np.zeros((patchsize*patchsize, patchsize*patchsize))
    flip = np.zeros((patchsize*patchsize, patchsize*patchsize))
    for i in range(0, patchsize*patchsize):
        i1 = i % patchsize
        i2 = floor(i / patchsize)
        j = patchsize * patchsize - patchsize + i2 - patchsize * i1
        rotate[j,i] = 1 
        k = patchsize * (i2 + 1) - i1 - 1
        flip[k,i] = 1
    for i in range(1, 8):
        i1 = i % 4
        i2 = floor(i / 4)
        P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
    return P

#@nb.jit(nopython=True, parallel=True) #JALALI中这步没有优化
def cal_qextend(R,Qangle,Qstrength,Qcoherence,P,Q,V,Qextended,Vextended):
    for pixeltype in range(0, R*R):
        for angle in range(0, Qangle):
            for strength in range(0, Qstrength):
                for coherence in range(0, Qcoherence):
                    for m in range(1, 8):
                        m1 = m % 4
                        m2 = floor(m / 4)
                        newangleslot = angle
                        if m2 == 1:
                            newangleslot = Qangle-angle-1
                        newangleslot = int(newangleslot-Qangle/2*m1)
                        while newangleslot < 0:
                            newangleslot += Qangle
                        newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                        newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
                        Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                        Vextended[newangleslot,strength,coherence,pixeltype] += newV
    return Qextended,Vextended

#@nb.jit(nopython=True, parallel=True)
def cal_h(R,Qangle,Qstrength,Qcoherence,h,Q,V):
    for pixeltype in range(0, R*R):
        for angle in range(0, Qangle):
            for strength in range(0, Qstrength):
                for coherence in range(0, Qcoherence):
                    h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])
    return h

@nb.jit(nopython=True,parallel=True)
def GetStrCoh(gx,gy,W):
    gx = gx.ravel()
    gy = gy.ravel()
    # SVD calculation
    G = np.vstack((gx,gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG)
    w = np.real(w)
    v = np.real(v)
    # Sort w and v according to the descending order of w
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    lamda = w[0]
    # Calculate u
    sqrtlamda1 = np.sqrt(w[0])
    sqrtlamda2 = np.sqrt(w[1])
    u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2 + 0.00000000000000001)
    return lamda,u

@nb.jit(nopython=True,parallel=True)
def hashkeyJIT(patchX,patchY, Qangle, W,strbin,cohbin):
    # Transform 2D matrix into 1D array
    gx = patchX.ravel()
    gy = patchY.ravel()
    # SVD calculation
    G = np.vstack((gx,gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG)
    w = np.real(w)
    v = np.real(v)
    # Sort w and v according to the descending order of w
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    # Calculate theta
    theta = atan2(v[1,0], v[0,0])
    if theta < 0:
        theta = theta + pi

    # Calculate lamda
    lamda = w[0]

    # Calculate u
    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001) #Junadd: equ(9)
    if isnan(u):
        u=1
    # Quantize
    angle = floor(theta/pi*Qangle)

    if lamda < strbin[0]:
        strength = 0
    elif lamda > strbin[1]:
        strength = 2
    else:
        strength = 1
    if u < cohbin[0]:
        coherence = 0
    elif u > cohbin[1]:
        coherence = 2
    else:
        coherence = 1

    # Bound the output to the desired ranges
    if angle > Qangle-1:
        angle = Qangle-1
    elif angle < 0:
        angle = 0

    return angle, strength, coherence

def save3DASCII(arr, pth):
    with open(pth, 'wb+') as outfile:
        outfile.write('{0:} {1:} {2:}\n'.format(arr.shape[0], arr.shape[1], arr.shape[2]).encode('ascii'))
        for slice_2d in arr:
            np.savetxt(outfile, slice_2d, delimiter=' ', fmt='%-7f')

def saveASCII_F(arr, pth):
    with open(pth, 'wb+') as outfile:
        np.savetxt(outfile, arr, delimiter=' ', fmt='%-7f')
