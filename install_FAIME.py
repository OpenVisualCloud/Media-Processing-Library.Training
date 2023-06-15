# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python

import os
import subprocess
import shutil
def getCmdOut(cmd):
    results = []
    res = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in iter( res.stdout.readline, b''):
        line_str = line.decode().strip()
        results.append(line_str)
    res.wait()
    res.stdout.close()
    return results

req_files = ['requirements.txt']

# Install wheel
subprocess.run(['pip','install','wheel'])

#Change execution mode for vmaf&gmaf
subprocess.run(['chmod','777','./FAIME/vmaf/vmaf'])
subprocess.run(['chmod','-R','777','./FAIME/gmaf/'])

# Install FFmpeg
# subprocess.run(['sudo','apt','install','ffmpeg','-y'])
# Install fonts-freefont-ttf
systemInfo = (' ').join(getCmdOut("lsb_release -a")).lower()
if 'centos' in systemInfo:
    subprocess.run(['sudo','yum','install','gnu-free-sans-fonts'])
elif 'ubuntu' in systemInfo:
    subprocess.run(['sudo','apt-get','install','-y','fonts-freefont-ttf'])
else:
    print("Nonsupport System")
#Copy ttf file into project
font_path = getCmdOut("fc-list| grep FreeSans.ttf")[0].split(':')[0]
database_font = os.path.join('FAIME','resources')
webpage_font = os.path.join('FAIME','visual_quality_tool','fonts')
if not os.path.exists(database_font):
    os.makedirs(database_font)
if not os.path.exists(webpage_font):
    os.makedirs(webpage_font)
shutil.copy(font_path, os.path.join(database_font,'FreeSans.ttf'))
shutil.copy(font_path, os.path.join(webpage_font,'FreeSans.ttf'))
# Install packages from requirements.txt
for fi in req_files:
    subprocess.run(['pip', 'install', '-r', fi])

# Install the current directory as an editable python package
subprocess.run(['pip', 'install', '-e', '.'])

# Install any packages in the 'packages' folder
for package in os.listdir('packages'):
    subprocess.run(['pip', 'install', '-e', os.path.join('packages', package)])

# Install Haarpsi
subprocess.run(['git','submodule','init'])
subprocess.run(['git','submodule','update'])
os.chdir('./haarpsi')
subprocess.run(['git','checkout','2079e05b730e6f8a1880c5bb6770d8bab10997b8'])