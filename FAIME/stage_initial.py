# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
from os import path
from stage import Stage, StageEnum

STAGE_NAME = 'StageInitial'

class StageInitial(Stage):

	def __init__(self):
		self.enum = StageEnum.INITIAL
		pass

	def ExecuteStage(self):
		logging.info("StageInital..executing stage")

		# make sure project folder exists
		if settings.set.project_folder != "":
			if not path.exists(settings.set.project_folder):
				logging.warning("WARNING =..project folder does not exist: "+settings.set.project_folder)
				quit()