# BanzaiBeach Pre-production Release Notes

**Release Version 23.3**
##   Release Information
This release includes components mainly related to Raisr only (Dataset preparation, training, testing, metrics calculation, comparison, visualization) and also supports default algorithms like bicubic, bilinear, lanczos for reference.
See README for more details.
## Main Features
- Super Resolution algorithm dataset building, from both images and videos.
- RAISR (Rapid and Accurate Image Super Resolution) filter training, and testing.
- For comparison algorithms, it enables CV algorithms including bicubic, bilinear, and lanczos.
- It supports image quality metrics including SSIM, MS-SSIM, PSNR, VMAF, GMAF, and HAARPSI and supports logging and plot generation.
- Results comparison and visualization are supported to provide easy-to-use interfaces for users.
- It is a flexible framework that supports single-stage execution or multi-stage execution of the pipeline.
- This framework has specific support for both 8-bit & 10-bit images/videos.

## Known Limitations
- Only mp4 videos are supported when using videos to build a dataset.
- Only image formats like png/jpg, etc. Can't support y4m/yuv files.
- Only Raisr training is supported in the training part.
- Downscale algorithm index have malposition, please set `downscale_algorithm` parameter in the correct index order when do downscaling:
    - 0=nearest, 1=bilinear, 2=bicubic, 3=area, 4=Lanczos, 5=Blur, 6=Random
## Related Documentation
- README.md
- ReleaseNotes.md

## List of Release Files
- config_files\ - config files of different situations. See README.
- FAMIE\ - contains .py source codes and files required for each stage of BanzaiBeach.
- misc\ - contains compare_regions.txt that may be used in the comparison stage.
- packages\ - contains source codes about Raisr training and Haarpsi metric calculation.
- FAIME.py - .py file to enable FAIME package in the working directory.

- install_FAIME.py - .py file to install necessary packages.
- install.sh - .sh file to prepare the environment for BanzaiBeach including creating a virtual environment and using install_FAIME.py to install packages.
- requirments.txt - .txt file contains all necessary python components.
- setup.cfg - .cfg file to enable FAIME package installation.
- setup.py - .py file to enable FAIME package installation.
- setup.sh - .sh file to set paths. See README.
## Acronyms and Terms
- **Banzai Beach**

    This beach is best known for the Banzai pipeline, a challenging series of barreling waves for only the best surfers. Given the pipeline architecture of this framework, it is an appropriate name choice.
- **Raisr**

    Rapid and Accurate Image Super Resolution
