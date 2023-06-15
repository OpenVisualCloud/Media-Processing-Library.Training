# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# abstract base class
from abc import ABC, abstractmethod
from enum import IntEnum

class StageEnum(IntEnum):
	INITIAL     = 0
	DOWNLOAD    = 1
	SCENEDETECT = 2
	CLASSIFY    = 3
	SPLIT       = 4
	CROP        = 5
	DOWNSCALE   = 6
	AUGMENT     = 7
	TRAIN       = 8
	TEST        = 9
	METRICS     = 10
	LOG         = 11
	COMPARE     = 12
	PLOT        = 13
	FINAL       = 14
		
class Stage:

	def __init__(self):
		pass

	def SetStageEnum(self, en):
		self.stageenum = en
		
	#@abstractmethod
	def StageEnm(self):
		return self.stageenum
	
	@abstractmethod
	def PrepareStage(self):
		pass
		
	@abstractmethod
	def ExecuteStage(self):
		pass
			
	@abstractmethod
	def CleanStage(self):
		pass
