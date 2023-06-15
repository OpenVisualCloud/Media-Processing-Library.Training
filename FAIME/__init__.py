# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from .FAIME import FAIME, main
from .project_builder import project_builder as ProjectBuilder
import video
del os
del sys
# __all__ = ['vsrframework']