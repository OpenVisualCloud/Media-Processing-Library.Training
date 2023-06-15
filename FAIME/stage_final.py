# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
from stage import Stage, StageEnum

STAGE_NAME = 'StageFinal'

class StageFinal(Stage):

	def __init__(self):
		self.enum = StageEnum.FINAL
		pass

	def ExecuteStage(self):
		logging.info("StageFinal..executing stage")
		