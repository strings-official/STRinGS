#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
import time

strings_360_scenes = ["extinguisher", "books", "chemicals", "globe", "shelf"]
strings_360_max_densify = [25, 25, 25, 25, 25]
tanks_and_temples_scenes = ["truck", "train"]
tanks_and_temples_max_densify = [20, 15]
dl3dv_scenes = ["scene_3", "scene_21", "scene_80", "scene_92", "scene_107", "scene_132", "scene_136"]
dl3dv_max_densify = [20, 20, 15, 20, 20, 15, 15]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_expcomp", action="store_true")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--aa", action="store_true")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(strings_360_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(dl3dv_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--strings360', "-s360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--dl3dv", "-dl3dv", required=True, type=str)
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 "
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_depth:
        common_args += " -d depths2/ "

    if args.use_expcomp:
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    start_time = time.time()
    for scene, max_densify in zip(strings_360_scenes, strings_360_max_densify):
        source = args.strings360 + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + " --max_densify " + str(max_densify) + common_args)
    s360_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene, max_densify in zip(tanks_and_temples_scenes, tanks_and_temples_max_densify):
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + " --max_densify " + str(max_densify) + common_args)
    tandt_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene, max_densify in zip(dl3dv_scenes, dl3dv_max_densify):
        source = args.dl3dv + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + " --max_densify " + str(max_densify) + common_args)
    dl3dv_timing = (time.time() - start_time)/60.0

with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
    file.write(f"s360: {s360_timing} minutes \n tandt: {tandt_timing} minutes \n dl3dv: {dl3dv_timing} minutes\n")

if not args.skip_rendering:
    all_sources = []
    for scene in strings_360_scenes:
        all_sources.append(args.strings360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in dl3dv_scenes:
        all_sources.append(args.dl3dv + "/" + scene)
    
    common_args = " --quiet --eval --skip_train"
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_expcomp:
        common_args += " --train_test_exp "

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
    os.system("python metrics_ocr.py -m " + scenes_string)
