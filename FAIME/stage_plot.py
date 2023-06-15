# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import settings
import database
import os
import PIL.Image
import glob
from stage import Stage, StageEnum
import math
import utils

STAGE_NAME = 'StagePlot'

class StagePlot(Stage):

    def __init__(self):
        self.enum = StageEnum.PLOT
        pass

    def ExecuteStage(self):
        logging.info("StagePlot..executing stage")

        if settings.set.plot_skip_stage:
            logging.info("StagePlot..skipping stage")
            return
        import matplotlib.pyplot as plt
        import numpy as np

        # get the number of scenes
        numscenes = database.db.getNumScenes()

        # set a low warning level for matplotlib
        plt.set_loglevel('WARNING')

        # make sure the plots folder exists
        if not os.path.exists(settings.set.absolute_plots_folder):
            os.mkdir(settings.set.absolute_plots_folder)

        # a list of thumbnails to draw in the plot
        thumbnails = []

        # for each metric to plot
        for metric in settings.set.plot_metrics:

            # x axis labels (scene #s)
            xlabels = []

            # create  dictionary of values for this plot
            values = {}

            # for each scene
            sceneindices = database.db.getSceneIndices()
            for sceneid in sceneindices :
                scene = database.db.getScene(sceneid)

                # append the scene label
                #xlabels.append(str(scene['scene_index']) + "-" +scene['video_name'] + "(" + str(scene['start_frame']) + '-' + str(scene['end_frame']) + ")")
                xlabels.append(str(scene['scene_index']) + " " + scene['folder_name'])

                # for each test recorded for the scene
                for test_name in scene['tests']:

                    # see if we are plotting for one specific test
                    if settings.set.plot_test_name != "" and test_name != settings.set.plot_test_name:
                        continue

                    # build a dictionary with the values to plot
                    if test_name not in values.keys():
                        values[test_name] = []

                    # get the metrics associated with this scene and test
                    metrics_dict = scene["tests"][test_name]
                    if settings.set.plot_test_name != "" and settings.set.plot_test_name not in scene["tests"]:
                        logging.warning(
                            "WARNING: test not found for scene " + str(scene['scene_index']) + " test: " + settings.set.log_test_name)
                        values[test_name].append(0.0)
                        continue

                    # see if the metric is defined for this scene-test
                    if metric not in metrics_dict:
                        logging.warning("WARNING: metric not found for scene and test " + str(scene['scene_index']) + " test: " + test_name + " metric: "+ metric)
                        values[test_name].append(0.0)
                        continue

                    # get the actual value
                    value = metrics_dict[metric]

                    # special case for infinity
                    if value != math.inf:
                        values[test_name].append(value)
                    else:
                        values[test_name].append(0.0)

                #  thumbnails
                if metric == settings.set.plot_metrics[0]:
                    hrfiles = utils.GetFiles(scene['hr_path'])
                    thumbnails.append(PIL.Image.open(hrfiles[0]))

            # make sure all
            # how mny bars to plot (one per test)
            num_bars = len(values)
            if num_bars == 0:
                logging.warning("WARNING: no data found to plot.")
                return

            # the label locations (one per scene)
            x = np.arange(len(xlabels))

            # the width of a bar (leave some blank space)
            barwidth = 1.0/ (num_bars+1)

            # get the figure and axis
            fig, ax = plt.subplots()

            # Create the bars for the plot
            for idx, key in enumerate(values):
                if key in values:
                    # plot the bars
                    barcenter = x-idx*barwidth + (num_bars-1)*barwidth*0.5
                    if idx==0:
                        numScenes = len(values[key])
                    else:
                        if len(values[key]) != numScenes:
                            logging.warning("WARNING..Metrics not computed for all scenes")
                            continue

                    ax.bar(barcenter, values[key], barwidth, label=key)


                    # plot values on top of the bars (3 decimal places of precision)
                    #for j, v in enumerate(values[key]):
                    #    # don't plot out 0 values
                    #    if v != 0.0:
                    #        plt.text(barcenter[j], v + 0.02, "{:.3f}".format(v), fontsize=5,  horizontalalignment='center')
                else:
                    logging.warning("WARNING: unable to plot.  Values not found for "+ key)

            # y axis labels and title (x axis label not needed)
            plt.ylabel(metric)
            plt.title(metric + ' vs Scene')

            # grid lines - horizontal only
            plt.grid(axis='y')

            # Scene description below the x axis and rotated
            plt.xticks(x, xlabels)
            plt.xticks(rotation=90)
            ax.tick_params(axis='x', which='major', pad=45)

            # legend
            plt.legend(bbox_to_anchor=(1.05, 1.0), fancybox=True, shadow=True, loc="upper left", title="Algorithm")

            # leave some space between the plot elements
            plt.tight_layout()

            # get the figure size in pixels (should be 640x480)
            figurewidth  = fig.get_figwidth() * fig.dpi
            figureheight = fig.get_figheight() * fig.dpi

            # get the y axis limits
            ymin, _ = ax.get_ylim()

            # get the tickmarks in pixel positions
            at = ax.transData.transform([(xtick, ymin) for xtick in ax.get_xticks()])

            # get the x-axis y percent.  Subtract a few pixels to allow the x axis to fully show
            x_axis_y_percent = float(ax.bbox.y0-2)/float(figureheight)

            # determine the spacing between x axis tick marks
            if len(ax.get_xticks())==1:
                x_axis_spacing_percent = 0.2
            else:
                x_axis_spacing_percent = (at[1][0]-at[0][0])/float(figurewidth)

            # The thumbnail width will be a percentage of the spacing
            thumbnail_width_percent = x_axis_spacing_percent*0.85

            # For each thumbnail
            for idx, im in enumerate(thumbnails):

                # determine the aspect ratio of the thumbnail
                thumbnail_aspect_ratio = float(im.height)/float(im.width)
                thumbnail_height_percent = thumbnail_width_percent * thumbnail_aspect_ratio * fig.get_figwidth() /fig.get_figheight()

                # determine the bounding box of the thumbnail within the figure
                thumbnail_boundingbox = [at[idx][0]/float(figurewidth) - thumbnail_width_percent*0.5, x_axis_y_percent-thumbnail_height_percent, thumbnail_width_percent, thumbnail_height_percent]

                # add an axes to contain the thumbnail
                ax1 = fig.add_axes(thumbnail_boundingbox)

                # make the x and y axis on the thumbnail axes invisitble
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)

                # show the thumbnail axes -stretch the image to fill the axes size
                ax1.imshow(im, interpolation='nearest', aspect='auto')

            # save the plot figure
            plotfilepath = os.path.join(settings.set.absolute_plots_folder, metric + ".png")
            fig.savefig(plotfilepath)
            plt.close(1)

        logging.info("StagePlot..stage finished")