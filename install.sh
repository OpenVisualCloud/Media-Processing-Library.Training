#!/usr/bin/env bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Make sure python virtual environments are installed on your server
sudo apt-get install python3.7-venv
# Create the virtual environment
python3 -m venv venv
# Activate the virtual environment
source ./venv/bin/activate
# Upgrade pip (python installer)
pip3 install --upgrade pip
# Clear out your python pip cache:
rm -r ~/.cache/pip
# Upgrade pip's setup tools
pip3 install --upgrade setuptools
# Install requirements
python3 install_FAIME.py
