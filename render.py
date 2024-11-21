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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, focal_length):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx==1:
        ##if you want to render the AIF HDR, set the ap a very large value
        #view.ap = 100000000000000
        rendering = render(view, gaussians, pipeline, background, focal_length = focal_length)["render"]
        radii = render(view, gaussians, pipeline, background, focal_length = focal_length)["radii"]

        gt = view.original_image[0:3, :, :]
        
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        time = view.time



        image_tmo = torch.pow(torch.clamp((rendering*2**time), 0.0, 1.0)+ 0.0000000001, 0.5)
        torchvision.utils.save_image(image_tmo, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        rendering_np = rendering.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)

        cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".hdr"), cv2.cvtColor(rendering_np, cv2.COLOR_RGB2BGR))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, focal_length):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train_aif_new", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, focal_length)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, focal_length)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.length_focal)