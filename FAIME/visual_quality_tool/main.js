// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

const defaultCellSize = 16;
const defaultScreenWidth = 2560;
const videoFPS = 30;
const chartColors = [
	"rgb(255, 99, 132)",
	"rgb(255, 159, 64)",
	"rgb(255, 205, 86)",
	"rgb(75, 192, 192)",
	"rgb(54, 162, 235)",
	"rgb(153, 102, 255)",
	"rgb(201, 203, 207)"
];

var lightThemeColors = {
    primaryColor: "#FFFFFF",
    secondaryColor: "#0068B5",
    textColor: "#000000",
    firstLevelColor: "#F4F5F5",
    secondLevelColor: "#E9EAEB",
    borderColor: "#C9CCD1",
    disabledColor: "#BAC1CE"
}

var darkThemeColors = {
    primaryColor: "#373C47",
    secondaryColor: "#00C7FD",
    textColor: "#B4B7C1",
    firstLevelColor: "#2C303B",
    secondLevelColor: "#292D37",
    borderColor: "#1c1c33",
    disabledColor: "#4A5568"
}

var cellSize;
var themeColors = lightThemeColors;

var sequences = [];
var magnifiers = [];
var magnifierImages = [];

var currentSequenceIndex = -1;
var currentModelIndex = -1;
var currentFrameIndex = -1;

var metricsAnnotations;
var metricsInterval = {
    start: -1,
    end: -1
}

var upscaleRatio = 1.0;
var magnifierZoom = 2.0;
var currentZoom = 1.0;
var oldZoom = 1.0;

var firstFrame = 0;
var lastFrame = 0;

var mainImage;
var diffImage;
var heatmapImage;

var selectedModel;
var imageHeight;
var imageWidth;

var cycleTimer;
var cyclePlay;

var fixedOffset;
var fixedScale;
var magnifierSize;

var darkModeToggled = false;
var fixedCursorPosition = false

var imageLoaded = false;
var imageUpdated = false;

var diffAvailable = false;
var diffEnabled = false;

var heatmapAvailable = false;
var heatmapEnabled = false;

var xCoord;
var yCoord;

var models = [
    $("#model-1"),
    $("#model-2"),
    $("#model-3"),
    $("#model-4"),
    $("#model-5"),
    $("#model-6"),
    $("#model-7"),
    $("#model-8"),
    $("#model-9")
];

var eyes = [
    $("#eye-1"),
    $("#eye-2"),
    $("#eye-3"),
    $("#eye-4"),
    $("#eye-5"),
    $("#eye-6"),
    $("#eye-7"),
    $("#eye-8"),
    $("#eye-9 ")
];

var controlButtons = {
    fullscreen: $("#fullscreen-button"),
    pauseVideo: $("#pause-video-button"),
    playVideo: $("#play-video-button"),
    diff: $("#diff-button"),
    heatmap: $("#heatmap-button"),
    firstFrame: $("#first-frame-button"),
    prevFrame: $("#prev-frame-button"),
    nextFrame: $("#next-frame-button"),
    lastFrame: $("#last-frame-button"),
    fitFrame: $("#fit-frame-button"),
    MagFlip: $("#magnifier-orientation-button"),
    helpButton: $("#help-button")
}

var mainBody = $("body");
var displayGrid = $("#display-grid");

var menuButton = $("#menu-button input");
var mainMenu = $("#main-menu");

var configSelector = $("#config-selector");
var configInput = $("#config-input");
var darkMode = $("#dark-mode");

var speedSelector = $("#speed-selector");
var frameSelector = $("#frame-selector");
var zoomSelector = $("#zoom-selector");
var magnifierZoomSelector = $("#magnifier-zoom-selector");

var frameStats = $("#frame-stats");

var cyclePlayButton = $("#cycle-play-button input");
var heatmapButton = $("#heatmap-button input");
var magOrientButton = $("#magnifier-orientation-button input");
var diffButton = $("#diff-button input");

var seqLabel = $("#seq-label input");
var seqButton = $("#seq-button input");
var seqMenu = $("#seq-menu");

var showChartButton = $("#show-chart-button");
var hideChartButton = $("#hide-chart-button");

var metricsLabel = $("#metrics-label input");
var metricsButton = $("#metrics-button input");
var metricsMenu = $("#metrics-menu");

var metricsGraph = $("#metrics-graph");
var metricsCanvas = $("#metrics-canvas");
var metricsOverlay = $("#metrics-overlay");
var metricsChart;

var imageWrapper = $("#image-wrapper");
var videoWrapper = $("#video-wrapper");

var imageArea = $("#image-area");
var videoArea = $("#video-area");
var magnifierArea = $("#magnifier-area");

var video = $("#video")[0];
var videoModeSwitch = $("#video-mode-switch");

var heatmapMenu = $("#heatmap-menu");
var heatmapOptions = $("#heatmap-options input");
var heatmapSlider = $("#heatmap-slider");

// Utils
function SetRootStyle(style, value) {
    document.documentElement.style.setProperty(style, value);
}

function CreateImage(imageId) {
    var image = document.createElement("img");
    image.id = imageId;
    return image;
}

function IsEmpty(obj) {
    for (var key in obj) {
        if (obj.hasOwnProperty(key))
            return false;
    }
    return true;
}

function Clamp(number, min, max) {
    return Math.max(min, Math.min(number, max));
  }

// Entry point
Initialization();

function ReadConfig(input) {
    var file = input.files[0];
    if (!file) {
        return;
    }

    console.log("Read config file " + file.name);

    var reader = new FileReader();

    reader.readAsText(file);
    reader.onload = function() {
        ParseConfig(JSON.parse(reader.result));
    };

    reader.onerror = function() {
        console.log(reader.error);
    };
}

function ParseConfig(config_data) {
    configData = config_data;

    sequences = []
    for (const seq of configData.sequences) {
        var sequence = {
            name: seq.name,
            firstFrame: seq.firstFrame,
            lastFrame: seq.lastFrame,
            models: []
        };
        for (const mdl of seq.models) {
            var model = {
                name: mdl.name,
                frames: mdl.frames,
                diffs: mdl.diffs,
                heatmaps: mdl.heatmaps,
                metrics: mdl.metrics,
                statistics: mdl.statistics,
                video: mdl.video
            };
            sequence.models.push(model);
        }
        if (!IsEmpty(seq.ref)) {
            sequence.ref = {
            frames: seq.ref.frames,
            statistics: seq.ref.statistics,
            video: seq.ref.video
            }
        }
        sequences.push(sequence);
    }

    UpdateConfigs();
}

function UpdateConfigs() {
    seqMenu.empty();
    currentSequenceIndex = -1;

    SetRootStyle("--num-sequences", sequences.length);
    seqLabel.next().text(sequences[0].name);

    for (const [i, seq] of sequences.entries()) {
        var option = $("<li/>").text(seq.name);
        option.on("click", function() {
            seqLabel.next().text(seq.name);
            seqLabel.prop("checked", false);
            seqLabel.trigger("change");
            UpdateSequence(i);
        });
        seqMenu.append(option);
    }

    UpdateSequence(0);
    imageUpdated = false;
}

function UpdateSequence(newSeqIndex) {
    if (currentSequenceIndex === newSeqIndex) {
        return;
    }
    console.log("Selected sequence " + seqLabel.next().text());

    for (const magnifier of magnifiers) {
        magnifier.title[0].remove();
        magnifier.magnifier[0].remove();
    }

    for (const image of magnifierImages) {
        image.remove();
    }

    for (var i = 0; i < models.length; i++) {
        models[i].find("div").text("");
        models[i].css("display", "none");
        eyes[i].css("display", "none");
    }

    currentSequenceIndex = newSeqIndex;
    currentModelIndex = -1;
    currentFrameIndex = 0;

    metricsInterval = {
        start: -1,
        end: -1
    }

    firstFrame = sequences[currentSequenceIndex].firstFrame;
    lastFrame = sequences[currentSequenceIndex].lastFrame;

    frameSelector.prop({
        "min": firstFrame,
        "max": lastFrame,
        "value": firstFrame
    });

    function CreateMagnifier(name, data) {
        function CreateMagnifierDiv(name) {
            var title = $("<div>").addClass("title");
            var label = $("<label>").addClass("container");
            label.text(name);

            var input = $("<input>");
            input.prop("type", "checkbox");

            var span = $("<span>").addClass("checkmark");

            label.append(input);
            label.append(span);
            title.append(label);

            var magnifier = $("<div>").addClass("magnifier");

            input.on("change", function() {
                magnifier.css("display", $(this).is(":checked") ? "block" : "none")
                title.css("display", $(this).is(":checked") ? "block" : "none")
            });

            input.prop("checked", !!data);
            input.trigger("change");

            return [title, magnifier, label];
        }

        var image = CreateImage(name + "-img");
        if (data) {
            image.src = data.frames[currentFrameIndex];
        }
        magnifierImages.push(image);
    
        [title, magnifier, label] = CreateMagnifierDiv(name);
        if (!data) {
            title.addClass("disabled-button");
        }

        magnifiers.push({
            title: title,
            magnifier: magnifier,
			name: name,
			label: label
        });
        var mgH = $("<div>").addClass("magHolder");
        magnifierArea.append(mgH);
        mgH.append(title);
        mgH.append(magnifier);
    }

    function UpdateModelTab(index, name, disabled = false) {
        models[index].find("div").text(name);
        models[index].find("div").prop("title", "[" + Number(index + 1) + "] " + name);
        models[index].css("display", "flex");
        eyes[index].css("display", "flex");
        eyes[index].find("input").prop("checked", false);

        if (disabled) {
            models[index].addClass("disabled-button");
            eyes[index].addClass("disabled-button");
        }
    }

    magnifiers = [];
    magnifierImages = [];

    var ref = IsEmpty(sequences[currentSequenceIndex].ref) ? null : sequences[currentSequenceIndex].ref;
    UpdateModelTab(0, "ref", ref == null);
    CreateMagnifier("ref", ref);

    for (const [i, mdl] of sequences[currentSequenceIndex].models.entries()) {
        UpdateModelTab(i + 1, mdl.name);
        CreateMagnifier(mdl.name, mdl);
    }

    UpdateModel(1);
    UpdateCycleState();
}

function UpdateModel(newModelIndex) {
    if (currentModelIndex === newModelIndex) {
        return;
    }

    currentModelIndex = newModelIndex;
    models[currentModelIndex].find("input").prop("checked", true);

    if (currentModelIndex === 0) {
        // Selected ref
        selectedModel = sequences[currentSequenceIndex].ref;
        console.log("Selected ref");

        if (diffEnabled) {
            diffButton.prop("checked", false);
            diffButton.trigger("change");
        }
        diffAvailable = false;

        if (heatmapEnabled) {
            heatmapButton.prop("checked", false);
            heatmapButton.trigger("change");
        }
        heatmapAvailable = false;
        heatmapButton.prop("disabled", true);
        heatmapOptions.prop("disabled", true);
    } else {
        // Selected model
        selectedModel = sequences[currentSequenceIndex].models[currentModelIndex - 1];
        console.log("Selected model " + selectedModel.name);

        diffAvailable = selectedModel.diffs.length !== 0;
        if (diffAvailable) {
            console.log("Diff available");
            diffImage.src = selectedModel.diffs[currentFrameIndex];
        }
        else if (diffEnabled) {
            diffButton.prop("checked", false);
            diffButton.trigger("change");
        }
    }
    diffButton.prop("disabled", !diffAvailable);

    mainImage.src = selectedModel.frames[currentFrameIndex];

    var isPlaying = !video.paused;
    var currentTime = video.currentTime;

    video.src = selectedModel.video;
    video.currentTime = currentTime;
    video.playbackRate = parseFloat(speedSelector.val());

    if (isPlaying) {
        // play() runs in async mode, it throws exceptions if interrupted
        video.play().catch((function(){}));
    }

    UpdateStatistics();
    if (currentModelIndex !== 0) {
        // Update metrics
        metricsMenu.empty();
        SetRootStyle("--num-metrics", Object.keys(selectedModel.metrics).length);

        for (let metrics in selectedModel.metrics) {
            var option = $("<li/>").text(metrics);
            option.on("click", function() {
                metricsLabel.next().text(metrics);
                metricsLabel.prop("checked", false);
                metricsLabel.trigger("change");
                UpdateChart();
            });
            metricsMenu.append(option);
        }

        metricsLabel.next().text($(":first-child", metricsMenu).text());
        UpdateChart();
    }
}

function UpdateStatistics() {
    function ProcessFloat(value) {
        return parseFloat(value.toFixed(3));
    }
    var statsText = "";
    if (!IsEmpty(selectedModel.statistics)) {
        var stats = selectedModel.statistics;
        statsText += "min: " + ProcessFloat(stats.min[currentFrameIndex]);
        statsText += " max: " + ProcessFloat(stats.max[currentFrameIndex]);
        statsText += " mean: " + ProcessFloat(stats.mean[currentFrameIndex]);
    }
    frameStats.text(statsText);
}

function MainImageLoaded() {
    console.log("Main image loaded");
    imageLoaded = true;

    UpdateImageInfo();
    UpdateImageSize();

    if (!diffEnabled && !heatmapEnabled) {
        imageArea.css("background-image", "url(" + mainImage.src + ")");
    }

    if (cyclePlay) {
        clearTimeout(cycleTimer);
        UpdateCycle();
    }
}

function ChangeFrame(new_frame_index, updateVideo = true) {
    if (new_frame_index < 0 || new_frame_index > lastFrame - firstFrame) {
        return;
    }

    currentFrameIndex = new_frame_index;
    frameSelector.val(currentFrameIndex + firstFrame);

    console.log("Current frame index: " + (currentFrameIndex + firstFrame));

    metricsAnnotations.currentFrameAnnotation.xMin = currentFrameIndex;
    metricsAnnotations.currentFrameAnnotation.xMax = currentFrameIndex;
    metricsChart.update();

    UpdateStatistics();

    // Update image panel
    mainImage.src = selectedModel.frames[currentFrameIndex];

    // Update diff
    if (diffAvailable) {
        diffImage.src = selectedModel.diffs[currentFrameIndex];
    }

    // Update heatmap
    if (heatmapAvailable) {
        heatmapImage.src = selectedModel.heatmaps[metricsLabel.next().text()][currentFrameIndex];
    }

    // Update image src for magnifiers
    if (!cyclePlay) {
        if (!IsEmpty(sequences[currentSequenceIndex].ref)) {
            magnifierImages[0].src = sequences[currentSequenceIndex].ref.frames[currentFrameIndex];
        }
        UpdateMagnifierDiffs();
    }

    // Update video
    if (updateVideo) {
        video.currentTime = currentFrameIndex / videoFPS;
    }

    imageLoaded = false;
}

function UpdateChart() {
    console.log("Update chart");

    var frameIds = []
    for (var i = 0; i < selectedModel.frames.length; i++) {
        frameIds.push(i + firstFrame);
    }

    function UpdateChartData() {
        function AddChartData(label, data, color) {
            var avgScore = data.reduce((a, b) => a + b) / data.length;
            metricsChart.data.datasets.push({
                label: label + " (" + avgScore.toFixed(3) + ")",
                data: data,
                fill: false,
                borderColor: color,
                backgroundColor: color
            })
        }

        var selectedMetrics = metricsLabel.next().text();
        console.log("Selected metrics: " + selectedMetrics);

        heatmapAvailable = selectedModel.heatmaps.hasOwnProperty(selectedMetrics);
        if (heatmapAvailable) {
            console.log(selectedMetrics + " heatmap available");
            heatmapImage.src = selectedModel.heatmaps[selectedMetrics][currentFrameIndex];
        }
        else if (heatmapEnabled) {
            heatmapButton.prop("checked", false);
            heatmapButton.trigger("change");
        }
        heatmapButton.prop("disabled", !heatmapAvailable);
        heatmapOptions.prop("disabled", !heatmapAvailable);

        metricsChart.data.labels = frameIds;
        metricsAnnotations.currentFrameAnnotation.xMin = currentFrameIndex;
        metricsAnnotations.currentFrameAnnotation.xMax = currentFrameIndex;

        metricsChart.data.datasets = [];
        for (const [i, mdl] of sequences[currentSequenceIndex].models.entries()) {
            if (mdl.metrics[selectedMetrics]) {
                AddChartData(mdl.name, mdl.metrics[selectedMetrics], chartColors[i]);
            }
        }
        metricsChart.resetZoom();
        metricsChart.update();
    }

    // Update chart
    if (metricsChart) {
        UpdateChartData();
        return;
    }

    metricsChart = new Chart(metricsCanvas[0].getContext("2d"), {
        type: "line",
        options: {
            animation: false,
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            scales: {
                x: {
                    display: true,
                    labelString: "Frame ID"
                },
                y: {
                    display: true,
                    labelString: "Score",
                    min: 0
                }
            },
            tooltips: {
                mode: "index",
                intersect: false,
                position: "nearest"
            },
            chartArea: {
                backgroundColor: themeColors.primaryColor
            },
            plugins: {
                annotation: {
                    annotations: {
                        currentFrameAnnotation: {
                            drawTime: "afterDatasetsDraw",
                            type: "line",
                            borderWidth: 1,
                            borderDash: [5, 5]
                        },
                        statrtIntervalAnnotation: {
                            drawTime: "afterDatasetsDraw",
                            type: "line",
                            display: false,
                            borderWidth: 3,
                            borderColor: "red",
                            borderDash: [5, 10]
                        },
                        endIntervalAnnotation: {
                            drawTime: "afterDatasetsDraw",
                            type: "line",
                            display: false,
                            borderWidth: 3,
                            borderColor: "red",
                            borderDash: [5, 10]
                        }
                    }
                },
                zoom: {
                    zoom: {
                      wheel: {
                        enabled: true,
                      },
                      mode: "y",
                    }
                }
            }
        }
    });

    Chart.register({
        id: "color_plugin",
        beforeDraw: function (chart) {
            if (chart.config.options.chartArea && chart.config.options.chartArea.backgroundColor) {
                var ctx = chart.ctx;
                var chartArea = chart.chartArea;

                ctx.save();
                ctx.fillStyle = chart.config.options.chartArea.backgroundColor;
                ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
                ctx.restore();
            }
        }
    });

    Chart.register({
        id: "overlay_plugin",
        afterLayout: function () {
            metricsOverlay[0].width = metricsCanvas[0].width;
            metricsOverlay[0].height = metricsCanvas[0].height;
            metricsOverlay.css({
                "width": metricsCanvas.css("width"),
                "height": metricsCanvas.css("height")
            });
        }
    });

    Chart.defaults.font.family = "FreeSans";
    Chart.defaults.font.size = cellSize;
    metricsAnnotations = metricsChart.options.plugins.annotation.annotations;

    UpdateChartData();
}

function UpdateZoom(newZoom) {
    var newZoom = parseFloat(newZoom);
    if (newZoom < zoomSelector.prop("min") || newZoom > zoomSelector.prop("max")) {
        return;
    }

    currentZoom = newZoom;
    zoomSelector.val(currentZoom.toFixed(1));

    UpdateImageSize();
}

function UpdateMagnifierZoom(newZoom) {
    var newZoom = parseFloat(newZoom);

    magnifierZoom = newZoom;
    magnifierZoomSelector.val(magnifierZoom.toFixed(1));

    UpdateFixedCursor();
    UpdateMagnifiers();
}


function SaveShot() {
	html2canvas(document.getElementById('magnifier-area')).then(function (canvas) {
		var anchorTag = document.createElement("a");
		document.body.appendChild(anchorTag);
		document.getElementById("magnifier-area").appendChild(canvas);
		anchorTag.download = Date.now().toString()+".png";
		anchorTag.href = canvas.toDataURL();
		anchorTag.target = '_blank';
		anchorTag.click();
	});
}

function UpdateFixedCursor() {
    if (fixedCursorPosition) {
        fixedOffset = magnifierZoom / currentZoom;
        fixedScale = magnifierZoom / upscaleRatio;
    }
}

function UpdateImageInfo() {
    if (imageUpdated) {
        return;
    }

    imageWidth = mainImage.naturalWidth;
    imageHeight = mainImage.naturalHeight;

    console.log("Main image resolution: " + imageWidth + "x" + imageHeight);
    imageArea.css({
        "height": imageHeight * currentZoom + "px",
        "width": imageWidth * currentZoom + "px"
    });

    imageUpdated = true;
}

function UpdateDiff() {
    var imageUrl = diffEnabled ?
        "url(" + diffImage.src + ")" :
        "url(" + mainImage.src + ")";

    imageArea.css({
        "background-image": imageUrl,
        "filter": ""
    });

    UpdateMagnifierDiffs();
}

function UpdateHeatmap() {
    var imageUrl = heatmapEnabled ?
        "url(" + mainImage.src + "), url(" + heatmapImage.src + ")" :
        "url(" + mainImage.src + ")";

    var blendMode = heatmapEnabled ? "luminosity" : "none";
    var filter = heatmapEnabled ? "saturate(" + heatmapSlider.val() + "%)" : ""; 

    imageArea.css({
        "background-image": imageUrl,
        "background-blend-mode": blendMode,
        "filter": filter
    });

    UpdateMagnifierDiffs();
}

function UpdateImageSize() {
    var scale = currentZoom / upscaleRatio;
    imageArea.css({
        "background-size": imageWidth * scale + "px " + imageHeight * scale + "px",
        "background-repeat": "no-repeat",
        "height": imageHeight * scale + "px",
        "width": imageWidth * scale + "px"
    });

    videoArea.css({
        "height": imageHeight * scale + "px",
        "width": imageWidth * scale + "px"
    });

    video.width = imageWidth * scale;
    video.height = imageHeight * scale;
}

function UpdateMagnifier(image, div, name, label) {
    var offset = fixedCursorPosition ? fixedOffset : magnifierZoom / currentZoom;
    var scale = fixedCursorPosition ? fixedScale : magnifierZoom / upscaleRatio;

    div.css({
        "background-image": "url(" + image.src + ")",
        "background-position": -xCoord * offset + "px " + -yCoord * offset + "px",
        "background-size": imageWidth * scale + "px " + imageHeight * scale + "px"
    });
	x = Math.floor(xCoord*offset)
	y = Math.floor(yCoord*offset)
	w = Math.floor(upscaleRatio*cellSize*16/magnifierZoom)
	h = Math.floor(upscaleRatio*cellSize*16/magnifierZoom)
	label.text(name+" x:" + x.toString() + " y:"+ y.toString() + " w:"+ w.toString()+ " h:"+ h.toString() );
}

function UpdateMagnifiers() {
    for (var i = 0; i < magnifiers.length; i++) {
         UpdateMagnifier(magnifierImages[i], magnifiers[i].magnifier, magnifiers[i].name, magnifiers[i].label);
    }
}

function UpdateMagnifierDiffs() {
    for (var i = 1; i < magnifierImages.length; i++) {
        var model = sequences[currentSequenceIndex].models[i - 1];
        magnifierImages[i].src = diffEnabled && model.diffs.length !== 0 ?
            model.diffs[currentFrameIndex] :
            model.frames[currentFrameIndex];
    }
    UpdateMagnifiers();
}

function UpdateCycle() {
    cycleTimer = setTimeout(function () {
        ChangeFrame(currentFrameIndex < metricsInterval.end - firstFrame ?
            currentFrameIndex + 1 :
            metricsInterval.start - firstFrame);
    }, 200);
}

function UpdateCycleState() {
    var resetCycle = metricsInterval.start === -1;

    if (resetCycle) {
        cyclePlayButton.prop("checked", false);
        cyclePlayButton.trigger("change");
    }
    cyclePlayButton.prop("disabled", resetCycle);

    if (!resetCycle) {
        metricsAnnotations.statrtIntervalAnnotation.xMin = metricsInterval.start - firstFrame;
        metricsAnnotations.statrtIntervalAnnotation.xMax = metricsInterval.start - firstFrame;

        metricsAnnotations.endIntervalAnnotation.xMin = metricsInterval.end - firstFrame;
        metricsAnnotations.endIntervalAnnotation.xMax = metricsInterval.end - firstFrame;
    }

    metricsAnnotations.statrtIntervalAnnotation.display = !resetCycle;
    metricsAnnotations.endIntervalAnnotation.display = !resetCycle;
    metricsChart.update();
}

function FitFrame() {
    imageArea.css({
        "left": 0 + "px",
        "top": 0 + "px"
    });

    videoArea.css({
        "left": 0 + "px",
        "top": 0 + "px"
    });

    UpdateZoom(imageWrapper[0].clientWidth / imageWidth);
    oldZoom = currentZoom;
}

function MagniFlip(value) {
	console.log("MagniFlip: "+value);
	if (magOrientButton.prop("checked") == false) {
		magOrientButton.prop("checked", true);
		magnifierArea.css("grid-column-start", 1);
		magnifierArea.css("grid-column-end", "span 160");
		magnifierArea.css("grid-row-end", 28);	}
	else {
		magOrientButton.prop("checked", false);
		magnifierArea.css("grid-column-start", 140);
		magnifierArea.css("grid-column-end", "span 21");
		magnifierArea.css("grid-row-end", 91);
	}
    magOrientButton.trigger("change");
}

function MouseMove(event) {
    if (cyclePlay || fixedCursorPosition) {
        return;
    }

    var rect =  event.target.getBoundingClientRect();
    var offset = magnifierSize * currentZoom / 2 / magnifierZoom;

    var x = event.clientX - rect.left - offset;
    var y = event.clientY - rect.top  - offset;

    var correctionFactor = currentZoom / upscaleRatio;

    xCoord = x < 0 ?
        0 : x > (imageWidth * correctionFactor - magnifierSize / 2) ? (imageWidth * correctionFactor) - (magnifierSize / 2) : x;
    yCoord = y < 0 ?
        0 : y > (imageHeight * correctionFactor - magnifierSize / 2) ? (imageHeight * correctionFactor) - (magnifierSize / 2) : y;

    UpdateMagnifiers();
}

function ToggleDarkMode() {
    themeColors = darkModeToggled ? darkThemeColors : lightThemeColors;
    SetRootStyle("--primary-color", themeColors.primaryColor);
    SetRootStyle("--secondary-color", themeColors.secondaryColor);
    SetRootStyle("--secondary-color-alpha", themeColors.secondaryColor + "75");
    SetRootStyle("--text-color", themeColors.textColor);
    SetRootStyle("--first-level-color", themeColors.firstLevelColor);
    SetRootStyle("--first-level-color-alpha", themeColors.firstLevelColor + "75");
    SetRootStyle("--second-level-color", themeColors.secondLevelColor);
    SetRootStyle("--second-level-color-alpha", themeColors.secondLevelColor + "75");
    SetRootStyle("--border-color", themeColors.borderColor);
    SetRootStyle("--disabled-color", themeColors.disabledColor);

    metricsGraph.css("background-color", themeColors.firstLevelColor);
    metricsCanvas.css("background-color", themeColors.firstLevelColor);
    if (metricsChart) {
        metricsChart.options.plugins.legend.labels.color = themeColors.textColor;
        metricsChart.options.scales.x.ticks.color = themeColors.textColor;
        metricsChart.options.scales.y.ticks.color = themeColors.textColor;
        metricsChart.options.scales.x.grid.color = themeColors.borderColor;
        metricsChart.options.scales.y.grid.color = themeColors.borderColor;
        metricsChart.options.chartArea.backgroundColor = themeColors.primaryColor;
        metricsAnnotations.currentFrameAnnotation.borderColor = themeColors.textColor;
        metricsChart.update();
    }
}

function AutoScale() {
    cellSize = (window.innerWidth / defaultScreenWidth) * defaultCellSize;
    var rowEnd = Math.min(Math.round(window.innerHeight / cellSize) + 1, 91);

    Chart.defaults.font.size = cellSize;
    magnifierSize = 16 * cellSize;

    SetRootStyle("--cell-size", cellSize + "px");
    SetRootStyle("--page-row-end", rowEnd);
}

function ShowHelp() {
    alert("Hotkeys:\n" +
        "[i]: fullscreen video\n" +
        "[o]: pause video\n" +
        "[o]: play video\n" +
        "[a]: first frame\n" +
        "[q]: previous frame\n" +
        "[e]: next frame\n" +
        "[d]: last frame\n" +
        "[f]: fit frame\n" +
        "[c]: cycle play\n" +
        "[x]: toggle heatmap\n" +
        "[x]: toggle diff\n" +
        "[g]: fixed cursor position\n" +
        "[1]-[9]: select model\n" +
        "[LMB]: pan image or video\n" +
        "[MouseWheel]: zoom image, video or chart\n" +
        "[Shift] + [o]: choose config\n" +
        "[Shift] + [d]: toggle dark mode\n" +
        "[Shift] + [v]: toggle video mode\n" +
        "Tips:\n" +
        "Click on eye on model tab to show/hide corresponded magnifier and chart line\n" +
        "Click on legend item on chart to show/hide corresponded chart line\n" +
        "Double click on data point on chart to change frame\n" +
        "Click and drag selection on chart to set cycle playback interval");
}

function Initialization() {
    upscaleRatio = window.devicePixelRatio;

    // Resize events
    window.addEventListener("resize", function() {
        upscaleRatio = window.devicePixelRatio;
        AutoScale();
        UpdateImageSize();
    });

    // Keyboard events
    document.addEventListener("keyup", function(event) {
        // Toggle grid hotkey
        if (event.altKey && event.key === "Enter") {
            var gridVisible = displayGrid.css("display") !== "none";
            displayGrid.css("display", gridVisible ? "none" : "block");
            gridVisible = !gridVisible;
            return;
        }

        // Select config hotkey
        if (event.shiftKey && event.key.toLowerCase() === "o") {
            configInput[0].click();
            return;
        }

        // Toggle theme hotkey
        if (event.shiftKey && event.key.toLowerCase() === "d") {
            darkModeToggled = !darkModeToggled;
            ToggleDarkMode();
            return;
        }

        // Video mode hotkey
        if (event.shiftKey && event.key.toLowerCase() === "v") {
            videoModeSwitch.click();
            return;
        }

        // Model tab hotkeys
        for (var i = 0; i < models.length; i++) {
            if (event.key === String(i) && !(
                frameSelector.is(":focus") ||
                speedSelector.is(":focus") ||
                zoomSelector.is(":focus") ||
                magnifierZoomSelector.is(":focus"))) {
                    models[i - 1].find("input").click();
                    return;
            }
        }

        // Button hotkeys
        switch(event.key.toLowerCase()) {
            case "i":
                controlButtons.fullscreen.click();
                break;
            case "o":
                controlButtons.pauseVideo.click();
                break;
            case "p":
                controlButtons.playVideo.click();
                break;
            case "q":
                controlButtons.prevFrame.click();
                break;
            case "e":
                controlButtons.nextFrame.click();
                break;
            case "a":
                controlButtons.firstFrame.click();
                break;
            case "d":
                controlButtons.lastFrame.click();
                break;
            case "f":
                controlButtons.fitFrame.click();
                break;
            case "c":
                cyclePlayButton.click();
                break;
            case "x":
                heatmapButton.click();
                break;
            case "y":
                SaveShot();
                break;
            case "z":
                diffButton.click();
                break;
            case "g":
                fixedCursorPosition = !fixedCursorPosition;
                UpdateFixedCursor();
                break;
        }
    });

    // Menu events
    menuButton.change(function() {
        mainMenu.css("display", $(this).prop("checked") ? "block" : "none");
    });

    configSelector.on("click", function() {
        mainMenu.css("display", "none");
        menuButton.prop("checked", false);
        configInput[0].click();
    });

    darkMode.on("click", function() {
        darkModeToggled = !darkModeToggled;
        darkMode.text(darkModeToggled ? "Light theme" : "Dark theme");
        ToggleDarkMode();

        mainMenu.css("display", "none");
        menuButton.prop("checked", false);
    });

    configInput.on("change", function() {
        ReadConfig($(this)[0]);
    });

    // Model tab events
    for (let i = 0; i < models.length; i++) {
        models[i].find("input").on("click", function() {
            UpdateModel(i);
        });
        eyes[i].find("input").on("change", function() {
            var hideModel = $(this).prop("checked");

            var magnifierCheckbox = magnifiers[i].title.find("input");
            magnifierCheckbox.prop("checked", !hideModel);
            magnifierCheckbox.trigger("change");

            if (i !== 0) {
                var meta = metricsChart.getDatasetMeta(i - 1);
                meta.hidden = hideModel;
                metricsChart.update();
            }
        });
    }

    // Control events
    controlButtons.fullscreen.on("click", function() {
        video.requestFullscreen();
    });

    controlButtons.pauseVideo.on("click", function() {
        video.pause();
    });

    controlButtons.playVideo.on("click", function() {
        video.play();
    });

    controlButtons.firstFrame.on("click", function() {
        ChangeFrame(0);
    });

    controlButtons.prevFrame.on("click", function() {
        ChangeFrame(currentFrameIndex - 1);
    });

    controlButtons.nextFrame.on("click", function() {
        ChangeFrame(currentFrameIndex + 1);
    });

    controlButtons.lastFrame.on("click", function() {
        ChangeFrame(lastFrame - firstFrame);
    });

    controlButtons.fitFrame.on("click", function() {
        FitFrame();
    });

    controlButtons.MagFlip.on("click", function() {
        MagniFlip(0);
    });

    speedSelector.change(function() {
        video.playbackRate = parseFloat($(this).val());
    });

    frameSelector.change(function() {
        ChangeFrame(Clamp(Number($(this).val()), firstFrame, lastFrame) - firstFrame);
    });

    // Video events
    video.addEventListener("pause", function() {
        ChangeFrame(Math.floor(video.currentTime * videoFPS), false);
    });

    video.addEventListener("seeked", function() {
        if ($(this).is(":focus")) {
            ChangeFrame(Math.floor(video.currentTime * videoFPS), false);
        }
    });

    videoModeSwitch.on("change", function() {
        if (!$(this).is(":checked")) {
            controlButtons.fullscreen.addClass("disabled-button");
            controlButtons.pauseVideo.addClass("disabled-button");
            controlButtons.playVideo.addClass("disabled-button");
            speedSelector.prop("disabled", true);
            speedSelector.css("pointer-events", "none");
        } else {
            controlButtons.fullscreen.removeClass("disabled-button");
            controlButtons.pauseVideo.removeClass("disabled-button");
            controlButtons.playVideo.removeClass("disabled-button");
            speedSelector.prop("disabled", false);
            speedSelector.css("pointer-events", "all");
        }

        videoArea.css("display", $(this).is(":checked") ? "flex" : "none");
        videoWrapper.css("display", $(this).is(":checked") ? "block" : "none");

        imageArea.css("display", $(this).is(":checked") ? "none" : "flex");
        imageWrapper.css("display", $(this).is(":checked") ? "none" : "block");
    });
    videoModeSwitch.trigger("change");

    mainImage = CreateImage("main-image");
    mainImage.addEventListener("load", function() { 
        MainImageLoaded();
    });

    diffImage = CreateImage("diff-image");
    diffImage.addEventListener("load", function() {
        if (diffEnabled) {
            UpdateDiff();
        }
    });

    heatmapImage = CreateImage("heatmap-image");
    heatmapImage.addEventListener("load", function() {
        if (heatmapEnabled) {
            UpdateHeatmap();
        }
    });

    // Toggle events
    cyclePlayButton.on("change", function(){
        cyclePlay = $(this).prop("checked");
        if (cyclePlay) {
            ChangeFrame(metricsInterval.start - firstFrame);
        }
    });

    heatmapButton.on("change", function() {
        heatmapEnabled = $(this).prop("checked");
        if (diffButton.prop("checked")) {
            diffButton.prop("checked", false);
            diffEnabled = false;
        }
        UpdateHeatmap();
    });

    diffButton.on("change", function() {
        diffEnabled = $(this).prop("checked");
        if (heatmapButton.prop("checked")) {
            heatmapButton.prop("checked", false);
            heatmapEnabled = false;
        }
        UpdateDiff();
    });

    // Sequence menu events
    function SeqMenuToggled() {
        let isOpen = seqMenu.css("display") !== "none";
        seqMenu.css("display", isOpen ? "none" : "block");
    }

    seqLabel.on("change", function() {
        console.log("AAA");
        seqButton.prop("checked", seqLabel.prop("checked"));
        SeqMenuToggled();
    });

    seqButton.on("change", function() {
        seqLabel.prop("checked", seqButton.prop("checked"));
        SeqMenuToggled();
    });

    function ToggleChart(style) {
        metricsGraph.css("display", style);
        hideChartButton.css("display", style);
        metricsLabel.parent().css("display", style);
        metricsButton.parent().css("display", style);
    }

    showChartButton.on("click", function() {
        ToggleChart("flex");
    });

    hideChartButton.on("click", function() {
        ToggleChart("none");
    });

    // Metrics menu events
    function MetricsMenuToggled() {
        metricsMenu.css("display", metricsMenu.css("display") !== "none" ? "none" : "block");
    }

    metricsLabel.on("change", function() {
        metricsButton.prop("checked", metricsLabel.prop("checked"));
        MetricsMenuToggled();
    });

    metricsButton.on("change", function() {
        metricsLabel.prop("checked", metricsButton.prop("checked"));
        MetricsMenuToggled();
    });

    // Heatmap menu events
    heatmapOptions.on("change", function() {
        heatmapMenu.css("display", heatmapMenu.css("display") !== "none" ? "none" : "flex");
    });

    heatmapSlider.on("input", function() {
        if (heatmapEnabled) {
            imageArea.css("filter", "saturate(" + $(this).val() + "%)");
        }
    })

    // Image scroll events
    var scrolling = false;
    var scrollPosition = {};

    $([imageWrapper, videoWrapper]).each(function() {
        $(this).on("mousedown", function(event) {
            if (event.which !== 1) {
                event.preventDefault();
                return;
            }

            scrolling = true;
            scrollPosition = {
                top: imageArea.css("top"),
                left: imageArea.css("left"),
                x: event.clientX,
                y: event.clientY,
            };
            imageWrapper.css("cursor", 'grabbing');
            videoWrapper.css("cursor", 'grabbing');
        });
    });

    $([imageWrapper, videoWrapper]).each(function() {
        $(this).on("mousemove", function(event) {
            if (scrolling) {
                const dx = event.clientX - scrollPosition.x;
                const dy = event.clientY - scrollPosition.y;

                imageArea.css({
                    "left": parseFloat(scrollPosition.left) + dx + "px",
                    "top": parseFloat(scrollPosition.top) + dy + "px"
                });

                videoArea.css({
                    "left": parseFloat(scrollPosition.left) + dx + "px",
                    "top": parseFloat(scrollPosition.top) + dy + "px"
                });
            }
        });
    });

    $([imageArea, videoArea]).each(function() {
        $(this).on("mousemove", function(event) {
            if (!scrolling) {
                MouseMove(event);
            }
        });
    });

    $([imageWrapper, videoWrapper]).each(function() {
        $(this).on("mouseup mouseout", function(event) {
            scrolling = false;
            imageWrapper.css("cursor", 'grab');
            videoWrapper.css("cursor", 'grab');
        });
    });

    // Image zoom events
    oldZoom = currentZoom;
    $([imageWrapper, videoWrapper]).each(function() {
        $(this).on('wheel', function(event) {
            event.preventDefault();
            var zoomDelta = event.originalEvent.deltaY < 0 ? 0.1 : -0.1;
            UpdateZoom(Number(currentZoom + zoomDelta).toFixed(1));

            var ratio = currentZoom / oldZoom;
            oldZoom = currentZoom;

            var offsetX = event.pageX - (event.pageX - parseFloat(imageArea.css("left"))) * ratio;
            var offsetY = event.pageY - (event.pageY - parseFloat(imageArea.css("top"))) * ratio;

            imageArea.css({
                "left": offsetX,
                "top": offsetY
            });

            videoArea.css({
                "left": offsetX,
                "top": offsetY
            });
        });
    });

    zoomSelector.on("change", function() {
        UpdateZoom(Number($(this).val()).toFixed(1));
    });

    magnifierZoomSelector.on("change", function() {
        UpdateMagnifierZoom(Number($(this).val()).toFixed(1));
    });

    // Chart events
    metricsCanvas.on("dblclick", function(event) {
        const point = metricsChart.getElementsAtEventForMode(event, 'nearest', {
            intersect: true
        }, false);
        if (point.length === 0) {
            return;
        }

        ChangeFrame(metricsChart.data.labels[point[0].index] - firstFrame);
    });

    var selectionContext = metricsOverlay[0].getContext('2d');
    var selectionRect = { w: 0, startX: 0, startY: 0 };
    var startIndex = 0;
    var dragSelection = false;

    metricsCanvas.on("pointerdown", function(event) {
        event.preventDefault();

        if (event.which !== 1) {
            return;
        }

        const point = metricsChart.getElementsAtEventForMode(event, 'index', {
            intersect: false
        });
        if (point.length === 0) {
            return;
        }

        startIndex = point[0].index;
        const rect = metricsCanvas[0].getBoundingClientRect();
        selectionRect.startX = (event.clientX - rect.left) * upscaleRatio;
        selectionRect.startY = metricsChart.chartArea.top * upscaleRatio;
        dragSelection = true;
    });

    metricsCanvas.on("pointermove", function(event) {
        const rect = metricsCanvas[0].getBoundingClientRect();
        if (dragSelection) {
            const rect = metricsCanvas[0].getBoundingClientRect();
            selectionRect.width = (event.clientX - rect.left) * upscaleRatio - selectionRect.startX;
            selectionContext.globalAlpha = 0.5;
            selectionContext.clearRect(0, 0, metricsOverlay[0].width, metricsOverlay[0].height);
            selectionContext.fillStyle = themeColors.borderColor;
            selectionContext.globalAlpha = 0.25;
            selectionContext.fillRect(selectionRect.startX,
                selectionRect.startY,
                selectionRect.width,
                (metricsChart.chartArea.bottom - metricsChart.chartArea.top) * upscaleRatio);
        } else {
            selectionContext.clearRect(0, 0, metricsOverlay[0].width, metricsOverlay[0].height);

            var x = (event.clientX - rect.left) * upscaleRatio;
            var y = (event.clientY - rect.top) * upscaleRatio;

            if (x > metricsChart.chartArea.left && x < metricsChart.chartArea.right &&
                y > metricsChart.chartArea.top && y < metricsChart.chartArea.bottom) {
                selectionContext.fillStyle = themeColors.borderColor;
                selectionContext.globalAlpha = 0.5;
                selectionContext.fillRect(x,
                    metricsChart.chartArea.top * upscaleRatio,
                    1,
                    (metricsChart.chartArea.bottom - metricsChart.chartArea.top) * upscaleRatio);
            }
        }
    });

    metricsCanvas.on("pointerup", function(event) {
        if (event.which !== 1) {
            return;
        }
        dragSelection = false;

        const point = metricsChart.getElementsAtEventForMode(event, 'index', {
            intersect: false
        });
        if (point.length === 0) {
            return;
        }

        var start = metricsChart.data.labels[startIndex];
        var end = metricsChart.data.labels[point[0].index];

        if (start === end) {
            return;
        }

        selectionRect = { w: 0, startX: 0, startY: 0 };

        metricsInterval.start = Math.min(start, end);
        metricsInterval.end = Math.max(start, end);

        console.log("Metrics start interval: " + metricsInterval.start);
        console.log("Metrics end interval: " + metricsInterval.end);

        UpdateCycleState();
    });

    // Help events
    controlButtons.helpButton.on("click", function() {
        ShowHelp();
    });

    // Body events
    mainBody.on("click", function(event) {
        if (["INPUT", "SPAN"].includes(event.target.tagName)) {
            return;
        }

        if (mainMenu.css("display") !== "none") {
            mainMenu.css("display", "none");
            menuButton.prop("checked", false);
        }

        if (seqMenu.css("display") !== "none") {
            seqMenu.css("display", "none");
            seqLabel.prop("checked", false);
            seqButton.prop("checked", false);
        }

        if (heatmapMenu.css("display") !== "none") {
            heatmapMenu.css("display", "none");
            heatmapOptions.prop("checked", false);
        }

        if (metricsMenu.css("display") !== "none") {
            metricsMenu.css("display", "none");
            metricsLabel.prop("checked", false);
            metricsButton.prop("checked", false);
        }
    });

    // Auto tune
    AutoScale();
    ToggleDarkMode();

    // Check cache
    if (!IsEmpty(cache_config)) {
        ParseConfig(cache_config);
    }
}
