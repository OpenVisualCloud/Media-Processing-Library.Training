# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import cv2
#from scipy.misc import
import PIL
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
from math import atan2, floor, pi, ceil, isnan
import numba as nb
import os
from os import fsync, remove, path
from numpy import frombuffer
from numba import njit, prange
from inspect import _void
from Y4M import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.y4m', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

# Python opencv library (cv2) cv2.COLOR_BGR2YCrCb has different parameters with MATLAB color convertion.
# In order to have a fair comparison with the benchmark, we wrote these functions by ourselves.
def BGR2YCbCr(im):
    mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    offset = np.array([[[16, 128, 128]]])
    if im.dtype == 'uint8':
        mat = mat/255
        out = np.dot(im,mat) + offset
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat/255
        offset = offset/255
        out = np.dot(im, mat) + offset
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out

def YCbCr2BGR(im):
    mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    mat = np.linalg.inv(mat)
    offset = np.array([[[16, 128, 128]]])
    if im.dtype == 'uint8':
        mat = mat * 255
        out = np.dot((im - offset),mat)
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat * 255
        offset = offset/255
        out = np.dot((im - offset),mat)
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out


def BGR2YCbCr_CV(im):
    out = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    return out

def YCbCr2BGR_CV(im):
    out = cv2.cvtColor(im, cv2.COLOR_YCrCb2BGR)
    return out

def im2double(im, bits):
    if im.dtype == 'uint8':
        out = im.astype('float') / 255
    elif im.dtype == 'uint16' and bits == '10':
        out = im.astype('float') / 1023
    elif im.dtype == 'uint16' and bits == '16':
        out = im.astype('float') / 65535
    elif im.dtype == 'float':
        out = im
    else:
        assert False
    out = np.clip(out, 0, 1)
    return out

def Gaussian2d(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def modcrop(im,modulo):
    shape = im.shape
    size0 = shape[0] - shape[0] % modulo
    size1 = shape[1] - shape[1] % modulo
    if len(im.shape) == 2:
        out = im[0:size0, 0:size1]
    else:
        out = im[0:size0, 0:size1, :]
    return out

# downscale -> upscale
def Prepare(im, R):
    H, W = im.shape
    #imL = imresize(im, 1 / R, interp='bicubic')
    size_tmp = tuple((np.array((W,H)) / R).astype(int))
    imL = np.array(Image.fromarray(im).resize(size_tmp, PIL.Image.BILINEAR))

    # cv2.imwrite('Compressed.jpg', imL, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    # imL = cv2.imread('Compressed.jpg')
    # imL = imL[:,:,0]   # Optional: Compress the image
    #imL = imresize(imL, (H, W), interp='bicubic')
    imL = np.array(Image.fromarray(imL).resize((W,H), PIL.Image.BILINEAR))

    imL = im2double(imL)
    im_LR = imL
    return im_LR

# downscale -> upscale
def PrepareCV(im, R):
    H, W = im.shape
    size_tmp = tuple((np.array((W,H)) / R).astype(int))

    imL = cv2.resize(im,size_tmp,interpolation=cv2.INTER_LINEAR)
    imCheapUpscaled = cv2.resize(imL,(W,H),interpolation=cv2.INTER_LINEAR)
    return imCheapUpscaled

# upscale only
def CheapUpscale(im, R):
    H, W = im.shape
    imCheapUpscaled = cv2.resize(im,(W*R,H*R),interpolation=cv2.INTER_LINEAR)
    return imCheapUpscaled

def LoadImageEntry(image, bits, R):
    if bits == '10':
        parser = Reader(_void, verbose=False)
        with open(image, 'rb') as fh:
            data = fh.read()
            if not data:
                return
            y4mtuple = parser.decode(data)
        im_1d = np.frombuffer(y4mtuple[0], dtype=np.uint16)
        im = np.reshape(im_1d, (-1, int(y4mtuple[2]['W'])))
        # OutputFile = open("im.yuv",'wb')
        # im.tofile(OutputFile)
        # OutputFile.close()
    else:
        # load HR image
        im = LoadImage(image)
    im = modcrop(im, R)
    return im


# use with 8bit png
def LoadImage(image):
    #print(image)
    # without IMREAD_UNCHANGED, 16bits image will convert to 8bits
    #im_uint8 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    im_uint8 = cv2.imread(image)
    if (im_uint8 is None):
        print("load image fails! exit! If you are using -chf or -lf, filename needs to be same as -ohf")
        exit()
    if is_greyimage(im_uint8):
        im_uint8 = im_uint8[:,:,0]
    if len(im_uint8.shape)>2:
        im_ycbcr = BGR2YCbCr_CV(im_uint8)
        im = im_ycbcr[:,:,0]
    else:
        im = im_uint8
    return im

def GetPath(filename, newpath):
    # Split the path in head and tail pair
    extensions = ['png','jpg','y4m']
    for ext in extensions:
        fi, _ = os.path.splitext(os.path.basename(filename))
        if os.path.exists(os.path.join(newpath,'.'.join([fi,ext]))):
            return os.path.join(newpath,'.'.join([fi,ext]))
    path, file = os.path.split(filename)
    newfilepath = os.path.join(newpath, file)
    return newfilepath

def is_greyimage(im):
    x = abs(im[:,:,0]-im[:,:,1])
    y = np.linalg.norm(x)
    if y==0:
        return True
    else:
        return False

@nb.jit(nopython=True, parallel=True)
def Grad(patchX,patchY,weight):
    gx = patchX.ravel()
    gy = patchY.ravel()
    G = np.vstack((gx,gy)).T
    x0 = np.dot(G.T,weight)
    x = np.dot(x0, G)
    w,v = np.linalg.eig(x)
    index= w.argsort()[::-1]
    w = w[index]
    v = v[:,index]
    lamda = w[0]
    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001)
    return lamda,u

@nb.jit(nopython=True, parallel=True)
def HashTable(patchX,patchY,weight, Qangle,Qstrength,Qcoherence,stre,cohe):
    assert (len(stre)== Qstrength-1) and (len(cohe)==Qcoherence-1),"Quantization number should be equal"
    gx = patchX.ravel()
    gy = patchY.ravel()
    G = np.vstack((gx,gy)).T
    x0 = np.dot(G.T,weight)
    x = np.dot(x0, G) #Junadd: GTWG
    w,v = np.linalg.eig(x)
    index= w.argsort()[::-1]
    w = w[index]
    v = v[:,index]
    theta = atan2(v[1,0], v[0,0])  #Junadd: equ(8)
    if theta<0:
        theta = theta+pi
    theta = floor(theta/(pi/Qangle))
    lamda = w[0]
    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001) #Junadd: equ(9)
    if isnan(u):
        u=1
    if theta>Qangle-1:
        theta = Qangle-1
    if theta<0:
        theta = 0
    lamda = np.searchsorted(stre,lamda)   #Junadd: search for insertion index
    u = np.searchsorted(cohe,u)
    return theta,lamda,u

@nb.jit(nopython=True, parallel=True)
def Gaussian_Mul(x,y,wGaussian):
    result = np.zeros((x.shape[0], x.shape[1], y.shape[2]))
    for i in range(x.shape[0]):
        # inter = np.matmul(x[i], wGaussian)
        # result[i] = np.matmul(inter,y[i])
        xi = x[i]
        yi = y[i]
        inter = np.dot(x[i], wGaussian)
        # print(inter)
        result[i] = np.dot(inter, y[i])
        resulti=result[i]
        # print(resulti)
    return result

CTWindowSize=3
def CT_descriptor(im):
    H, W = im.shape
    Census = np.zeros((H, W))
    CT = np.zeros((H, W, CTWindowSize, CTWindowSize))
    C = np.int((CTWindowSize-1)/2)
    for i in range(C,H-C):
        for j in range(C, W-C):
            cen = 0
            for a in range(-C, C+1):
                for b in range(-C, C+1):
                    if not (a==0 and b==0):
                        if im[i+a, j+b] < im[i, j]:
                            cen += 1
                            CT[i, j, a+C,b+C] = 1
            Census[i, j] = cen
            # print("CT:", CT[i, j], ", cen:", cen)
    Census = Census/8
    return Census, CT

def Blending1(LR, HR):
    H,W = LR.shape
    H1,W1 = HR.shape
    assert H1==H and W1==W
    Census,CT = CT_descriptor(LR)
    # the larger the census(LCC), the more obvious this pixel, it is more likely to be on a high freq detail. So we should take more from HR.
    # the smaller the census, the more likely this pixel is not on a structure. It is low freq info. We should take more from LR.
    blending1 = Census*HR + (1 - Census)*LR
    return blending1

def Blending2(LR, HR):
    H,W = LR.shape
    H1,W1 = HR.shape
    assert H1==H and W1==W
    Census1, CT1 = CT_descriptor(LR)
    Census2, CT2 = CT_descriptor(HR)
    weight = np.zeros((H, W))
    x = np.zeros((CTWindowSize, CTWindowSize))
    for i in range(H):
        for j in range(W):
            x  = np.absolute(CT1[i,j]-CT2[i,j])
            weight[i, j] = x.sum()
    # print("weight.max() ", weight.max()) # almost max() is 8
    # saveASCII_D(weight, "weight.dat")
    weight = weight/weight.max()
    # x show the CT changes that RAISR algo to the pixel. the larger the x, the more changes to the structure, which should be avoid(holas).
    # so the larger the weight, the more likelyhood the holas.
    blending2 = weight * LR + (1 - weight) * HR
    return blending2

def Backprojection(LR, HR, maxIter):
    H, W = LR.shape
    H1, W1 = HR.shape
    w = Gaussian2d((5,5), 10)
    w = w**2
    w = w/sum(np.ravel(w))
    for i in range(maxIter):
        #im_L = imresize(HR, (H, W), interp='bicubic', mode='F')
        im_L = np.array(Image.fromarray(HR).resize((W,H), PIL.Image.LINEAR))

        imd = LR - im_L
        #im_d = imresize(imd, (H1, W1), interp='bicubic', mode='F')
        im_d = np.array(Image.fromarray(imd).resize((W1,H1), PIL.Image.LINEAR))

        HR = HR + convolve2d(im_d, w, 'same')
    return HR

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def Dog1(im):
    sigma = 0.85
    alpha = 1.414
    r = 15
    ksize = (3, 3)
    G1 = cv2.GaussianBlur(im, ksize, sigma)
    Ga1 = cv2.GaussianBlur(im, ksize, sigma*alpha)
    D1 = cv2.addWeighted(G1, 1+r, Ga1, -r, 0)

    G2 = cv2.GaussianBlur(Ga1, ksize, sigma)
    Ga2 = cv2.GaussianBlur(Ga1, ksize, sigma*alpha)
    D2 = cv2.addWeighted(G2, 1+r, Ga2, -r, 0)

    G3 = cv2.GaussianBlur(Ga2, ksize, sigma)
    Ga3 = cv2.GaussianBlur(Ga2, ksize, sigma * alpha)
    D3 = cv2.addWeighted(G3, 1+r, Ga3, -r, 0)

    B1 = Blending1(im, D3)
    B1 = Blending1(im, B1)
    B2 = Blending1(B1, D2)
    B2 = Blending1(im, B2)
    B3 = Blending1(B2, D1)
    B3 = Blending1(im, B3)

    output = B3

    return output

def Getfromsymmetry1(V, patchSize, t1, t2):
    V_sym = np.zeros((patchSize*patchSize,patchSize*patchSize))
    for i in range(1, patchSize*patchSize+1):
        for j in range(1, patchSize*patchSize+1):
            y1 = ceil(i/patchSize)
            x1 = i-(y1-1)*patchSize
            y2 = ceil(j/patchSize)
            x2 = j-(y2-1)*patchSize
            if (t1 == 1) and (t2 == 0):
                ig = patchSize * x1 + 1 - y1
                jg = patchSize * x2 + 1 - y2
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 2) and (t2 == 0):
                x = patchSize + 1 - x1
                y = patchSize + 1 - y1
                ig = (y - 1) * patchSize + x
                x = patchSize + 1 - x2
                y = patchSize + 1 - y2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 3) and (t2 == 0):
                x = y1
                y = patchSize + 1 - x1
                ig =(y - 1) * patchSize + x
                x = y2
                y = patchSize + 1 - x2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 0) and (t2 == 1):
                x = patchSize + 1 - x1
                y = y1
                ig =(y - 1) * patchSize + x
                x = patchSize + 1 - x2
                y = y2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 1) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = patchSize + 1 - y0
                y = x0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = patchSize + 1 - y0
                y = x0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 2) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = patchSize + 1 - x0
                y = patchSize + 1 - y0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = patchSize + 1 - x0
                y = patchSize + 1 - y0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 3) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = y0
                y = patchSize + 1 - x0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = y0
                y = patchSize + 1 - x0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            else:
                assert False
    return V_sym

def Getfromsymmetry2(V, patchSize, t1, t2):
    Vp = np.reshape(V, (patchSize, patchSize))
    V1 = np.rot90(Vp, t1)
    if t2 == 1:
        V1 = np.flip(V1, 1)
    V_sym = np.ravel(V1)
    return V_sym

# Quantization procedure to get the optimized strength and coherence boundaries
@nb.jit(nopython=True, parallel=True)
def QuantizationProcess (im_GX, im_GY,patchSize, patchNumber,w , quantization, totpatches, R):
    H, W = im_GX.shape
    for k in range(totpatches):
            i1 = np.random.randint(H-4*floor(patchSize/2)-R*R+1)+floor(patchSize/2)
            j1 = np.random.randint(W-4*floor(patchSize/2)-R*R+1)+floor(patchSize/2)
            for i2 in range(R):
                for j2 in range(R):
                    i3 = i1 + i2
                    j3 = j1 + j2
                    idx = (slice(i3,(i3+2*floor(patchSize/2)+1)),slice(j3,(j3+2*floor(patchSize/2)+1)))
                    patchX = im_GX[idx]
                    patchY = im_GY[idx]
                    strength, coherence = Grad(patchX, patchY, w)
                    #print(strength,coherence)
                    quantization[patchNumber, 0] = strength
                    quantization[patchNumber, 1] = coherence
                    patchNumber += 1
                    totpatches -= 1
                    if totpatches < 1:
                        return quantization, patchNumber
    return quantization, patchNumber

# Training procedure for each image
def TrainProcess (im_LR, im_HR, im_GX, im_GY,patchSize, w, Qangle, Qstrength,Qcoherence, stre, cohe, R, Q, V, mark, theta, lamda, u):
    H, W = im_HR.shape
    for i1 in prange(H-2*floor(patchSize/2)):
        for j1 in range(W-2*floor(patchSize/2)):
            idx1 = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
            patch = im_LR[idx1]
            patchX = im_GX[idx1]
            patchY = im_GY[idx1]
            tA,lA,uA=HashTable(patchX, patchY, w, Qangle, Qstrength,Qcoherence, stre, cohe)
            theta[i1][j1] = tA
            lamda[i1][j1] = lA
            u[i1][j1] = uA
            #print(patch1.shape, patchL.shape, Q.shape, b1.shape, b.shape, V.shape, mark.shape)
            #      121           1,121         4x216x121x121  121,1   121    4x216x122   4x216
    for k1 in prange(80):
        for i1 in range(H-2*floor(patchSize/2)):
            for j1 in range(W-2*floor(patchSize/2)):
                j = theta[i1][j1] * Qstrength * Qcoherence + lamda[i1][j1] * Qcoherence + u[i1][j1]
                jx = np.int(j)
                if(jx%80 == k1):
                    idx1 = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
                    patch = im_LR[idx1]
                    patch1 = patch.ravel()
                    patchL = patch1.reshape((1,patch1.size))
                    t = (i1 % R) * R +(j1 % R) #Junadd: pixel type
                    tx = np.int(t)
                    A = np.dot(patchL.T, patchL) #Junadd: ATA
                    Q[jx, tx] += A
                    p2 = im_HR[i1+floor(patchSize/2),j1+floor(patchSize/2)]
                    b1=patchL.T
                    b2 = b1.reshape((b1.size))
                    for k2 in range(b1.size):
                        b2[k2] = b2[k2] * p2  #Junadd: ATb
                    V[jx, tx] += b2
                    mark[jx, tx] = mark[jx, tx]+1
    return Q,V,mark

# Training procedure for each image (use numba.jit to speed up)
@nb.jit(nopython=True, parallel=True)
def TrainProcessJIT(im_LR, im_HR, im_GX, im_GY,patchSize, w, Qangle, Qstrength,Qcoherence, stre, cohe, R, Q, V, mark, theta, lamda, u):
    H, W = im_HR.shape
    for i1 in prange(H-2*floor(patchSize/2)):
        for j1 in range(W-2*floor(patchSize/2)):
            idx1 = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
            patch = im_LR[idx1]
            patchX = im_GX[idx1]
            patchY = im_GY[idx1]
            tA,lA,uA=HashTable(patchX, patchY, w, Qangle, Qstrength,Qcoherence, stre, cohe)
            theta[i1][j1] = tA
            lamda[i1][j1] = lA
            u[i1][j1] = uA
            #print(patch1.shape, patchL.shape, Q.shape, b1.shape, b.shape, V.shape, mark.shape)
            #      121           1,121         4x216x121x121  121,1   121    4x216x122   4x216
    for k1 in prange(80):
        for i1 in range(H-2*floor(patchSize/2)):
            for j1 in range(W-2*floor(patchSize/2)):
                j = theta[i1][j1] * Qstrength * Qcoherence + lamda[i1][j1] * Qcoherence + u[i1][j1]
                jx = np.int(j)
                if(jx%80 == k1):
                    idx1 = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
                    patch = im_LR[idx1]
                    patch1 = patch.ravel()
                    patchL = patch1.reshape((1,patch1.size))
                    t = (i1 % R) * R +(j1 % R) #Junadd: pixel type
                    tx = np.int(t)
                    A = np.dot(patchL.T, patchL) #Junadd: ATA
                    Q[jx, tx] += A
                    p2 = im_HR[i1+floor(patchSize/2),j1+floor(patchSize/2)]
                    b1=patchL.T
                    b2 = b1.reshape((b1.size))
                    for k2 in range(b1.size):
                        b2[k2] = b2[k2] * p2  #Junadd: ATb
                    V[jx, tx] += b2
                    mark[jx, tx] = mark[jx, tx]+1
    return Q,V,mark

def sync(fh):
	"""
	This makes sure data is written to disk, so that buffering doesn't influence the timings.
	"""
	fh.flush()
	fsync(fh.fileno())

#save bin file(same as pickle without header)
def save2D(arr, pth):
    with open(pth, 'wb+') as fh:
        fh.write('{0:} {1:} {2:}\n'.format(arr.dtype, arr.shape[0], arr.shape[1]).encode('ascii'))
        fh.write(arr.data)
        sync(fh)

def save3DBin(arr, pth):
    with open(pth, 'wb+') as fh:
        fh.write('{0:} {1:} {2:} {3:}\n'.format(arr.dtype, arr.shape[0], arr.shape[1], arr.shape[2]).encode('ascii'))
        fh.write(arr.data)
        sync(fh)

def saveASCII_F(arr, pth):
    with open(pth, 'wb+') as outfile:
        np.savetxt(outfile, arr, delimiter=' ', fmt='%-7f')

def saveASCII_D(arr, pth):
    with open(pth, 'wb+') as outfile:
        np.savetxt(outfile, arr, delimiter=' ', fmt='%d')

def save3DASCII(arr, pth):
    with open(pth, 'wb+') as outfile:
        outfile.write('{0:} {1:} {2:}\n'.format(arr.shape[0], arr.shape[1], arr.shape[2]).encode('ascii'))
        for slice_2d in arr:
            np.savetxt(outfile, slice_2d, delimiter=' ', fmt='%-7f')

def save2DASCII(arr, pth):
    with open(pth, 'wb+') as outfile:
        outfile.write('{0:} {1:} \n'.format(arr.shape[0], arr.shape[1]).encode('ascii'))
        for slice_2d in arr:
            np.savetxt(outfile, slice_2d, delimiter=' ', fmt='%d')

def load2D(pth):
    with open(pth, 'rb') as fh:
        header = fh.readline()
        data = fh.read()
        dtype, w, h = header.decode('ascii').strip().split()
        return frombuffer(data, dtype=dtype).reshape((int(w), int(h)))
