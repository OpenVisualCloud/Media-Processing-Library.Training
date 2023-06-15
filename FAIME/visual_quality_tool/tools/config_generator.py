# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import shutil

from pathlib import Path

METRICS = {'gmaf': 'GMAF',
           'vmaf': 'VMAF',
           'psnr': 'PSNR',
           'ssim': 'SSIM',
           'ms_ssim': 'MS_SSIM',
           'flip': 'FLIP',
           'haar_psi': 'HAAR_PSI'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate config for Visual Quality Tool')

    parser.add_argument('-d', '--data',
                        required=True,
                        help='Input data JSON')

    parser.add_argument('-a', '--absolute-paths',
                        action='store_true',
                        help='Use absolute paths for config')

    return parser.parse_args()


def parse_data(args):
    with open(Path(args.data)) as data_file:
        data_json = json.load(data_file)
        generate_config(data_json, args.absolute_paths)


def generate_config(data_json, absolute_paths):
    def process_frame(png_file):
        png_path = str(png_file.resolve()) if absolute_paths else \
            str(png_file.relative_to(input_dir).resolve())
        frame = {
            'frame': png_path
        }
        return frame

    input_dir = Path(data_json['path'])

    config_json = input_dir / 'config.json' if absolute_paths \
        else input_dir / 'config.js'
    config_file = open(config_json, 'w')

    config = {'sequences': []}

    for seq_idx, seq in enumerate(data_json['sequences']):
        seq_dir = input_dir / seq['name']

        config['sequences'].append({
            'name': seq['name'],
            'models': []
        })

        png_files = []
        for model_idx, model in enumerate(seq['models']):
            model_dir = seq_dir / model
            print('Processing ' + str(model_dir))

            video_file = model_dir / 'dist.mp4'
            video_path = ""
            if video_file.is_file():
                video_path = str(video_file.resolve()) if absolute_paths else \
                    str(video_file.relative_to(input_dir).resolve())

            config['sequences'][seq_idx]['models'].append({
                'name': model,
                'frames': [],
                'diffs': [],
                'heatmaps': {},
                'metrics': {},
                'statistics': {},
                'video': video_path
            })

            print('\tProcessing dist frames')

            png_dir = model_dir / 'dist-png'
            png_files = sorted(png_dir.glob('*.png'))
            for frame_idx, png_file in enumerate(png_files):
                frame = process_frame(png_file)
                config['sequences'][seq_idx]['models'][model_idx][
                    'frames'].append(frame)

            for metrics, _ in METRICS.items():
                metrics_json = model_dir / Path(metrics + '.json')
                if metrics_json.is_file():
                    print('\tProcessing ' + metrics + ' metrics')

                    with open(metrics_json) as metrics_file:
                        metrics_data = json.load(metrics_file)
                        metrics_scores = []
                        for data in metrics_data['frames']:
                            metrics_scores.append(data[METRICS[metrics] + '_score'])

                    config['sequences'][seq_idx]['models'][model_idx][
                        'metrics'][metrics] = metrics_scores

                heatmap_dir = model_dir / 'heatmaps' / metrics / 'dist-png'
                heatmap_files = sorted(heatmap_dir.glob('*.png'))
                if heatmap_files:
                    print('\tProcessing ' + metrics + ' heatmap')

                    heatmaps = []
                    for heatmap_file in heatmap_files:
                        heatmap_path = str(heatmap_file.resolve()) if absolute_paths \
                            else str(heatmap_file.relative_to(input_dir).resolve())
                        heatmaps.append(heatmap_path)

                    config['sequences'][seq_idx]['models'][model_idx][
                        'heatmaps'][metrics] = heatmaps

            diff_dir = model_dir / 'diffs' / 'dist-png'
            diff_files = sorted(diff_dir.glob('*.png'))
            if diff_files:
                print('\tProcessing diffs')

                diffs = []
                for diff_file in diff_files:
                    diff_path = str(diff_file.resolve()) if absolute_paths \
                        else str(diff_file.relative_to(input_dir).resolve())
                    diffs.append(diff_path)

                config['sequences'][seq_idx]['models'][model_idx]['diffs'] = diffs

            stats_json = model_dir / 'dist-stats.json'
            if stats_json.is_file():
                print('\tProcessing dist stats')
                with open(stats_json) as stats_file:
                    stats_data = json.load(stats_file)

                    max_stat = []
                    mean_stat = []
                    min_stat = []

                    for frame_idx, _ in enumerate(png_files):
                        max_stat.append(stats_data['frames'][frame_idx]['max'])
                        mean_stat.append(
                            stats_data['frames'][frame_idx]['mean'])
                        min_stat.append(stats_data['frames'][frame_idx]['min'])

                config['sequences'][seq_idx]['models'][model_idx]['statistics'] = {
                    'max': max_stat,
                    'mean': mean_stat,
                    'min': min_stat
                }

        config['sequences'][seq_idx]['firstFrame'] = int(
            png_files[0].stem.split("_")[-1])
        config['sequences'][seq_idx]['lastFrame'] = int(
            png_files[-1].stem.split("_")[-1])

        # Add ref data
        png_dir = seq_dir / 'ref-png'
        if png_dir.is_dir():
            png_files = sorted(png_dir.glob('*.png'))
            video_file = seq_dir / 'ref.mp4'
            video_path = ""
            if video_file.is_file():
                video_path = str(video_file.resolve()) if absolute_paths else \
                    str(video_file.relative_to(input_dir).resolve())

            config['sequences'][seq_idx]['ref'] = {
                'frames': [],
                'statistics': {},
                'video': video_path
            }

            print('\tProcessing ref frames')

            for frame_idx, png_file in enumerate(png_files):
                frame = process_frame(png_file)
                config['sequences'][seq_idx]['ref']['frames'].append(frame)

            stats_json = seq_dir / 'ref-stats.json'
            if stats_json.is_file():
                print('\tProcessing ref stats')
                with open(stats_json) as stats_file:
                    stats_data = json.load(stats_file)

                    max_stat = []
                    mean_stat = []
                    min_stat = []

                for frame_idx, _ in enumerate(png_files):
                    max_stat.append(stats_data['frames'][frame_idx]['max'])
                    mean_stat.append(
                        stats_data['frames'][frame_idx]['mean'])
                    min_stat.append(stats_data['frames'][frame_idx]['min'])

                config['sequences'][seq_idx]['ref']['statistics'] = {
                    'max': max_stat,
                    'mean': mean_stat,
                    'min': min_stat
                }

    if not absolute_paths:
        root_dir = Path(__file__).resolve().parents[1]

        shutil.copy(root_dir / "index.html", input_dir)
        shutil.copy(root_dir / "main.js", input_dir)
        shutil.copy(root_dir / "style.css", input_dir)

        config_str = "var cached_data = " + json.dumps(config, indent=4,
                                                       sort_keys=True)
        config_file.write(config_str)
    else:
        json.dump(config, config_file, indent=4, sort_keys=True)

    print('Config file generated: ' + str(config_json))
    config_file.close()

    return config


if __name__ == '__main__':
    parse_data(parse_args())
