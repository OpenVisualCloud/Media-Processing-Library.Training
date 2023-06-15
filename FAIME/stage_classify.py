# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
import database
import multiprocessing as mp
import copy
from stage import Stage, StageEnum

STAGE_NAME = 'StageClassify'

class StageClassify(Stage):

	def __init__(self):
		self.enum = StageEnum.CLASSIFY
		pass

	def classify(self, scene, set):
		scene['motion'] = 0.0
		scene['texture_complexity'] = 0.0
		pass

	def ExecuteStage(self):
		logging.info("StageClassify.executing stage")

		if settings.set.classify_skip_stage:
			logging.info("StageClassify..skipping stage")
			return

		if len(database.db.scenes)== 0:
			logging.warning("WARNING no scenes found to classify")
			return

		sceneindices = database.db.getSceneIndices()
		if settings.set.multiprocess:
			processes = []

			logging.info("StageClassify..starting multiprocess..# of scenes = " + str(len(sceneindices)))

			dbcopy = copy.deepcopy(database.db)
			setcopy = copy.deepcopy(settings.set)
			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageClassify..starting process for scene=" + str(sceneid))

				p = mp.Process(target=self.classify, args=(database.db.getScene(sceneid), setcopy))
				processes.append(p)

			[p.start() for p in processes]
			[p.join() for p in processes]

			# copy back
			settings.set = copy.deepcopy(setcopy)
			database.db = copy.deepcopy(dbcopy)

			logging.info("StageSplit..multiprocess complete")

		else:
			logging.info("StageClassify..starting multiprocess..# of scenes = "  + str(len(sceneindices)))

			# for each scene:
			for sceneid in sceneindices:

				logging.info("StageClassify.starting process for scene=" + str(sceneid))

				self.classify(database.db.getScene(sceneid), settings.set)
