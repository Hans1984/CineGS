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
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, focal_length):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path_fd = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_fd")
    render_path_ap = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ap")
    render_path_exp = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_exp")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_path_fd, exist_ok=True)
    makedirs(render_path_ap, exist_ok=True)
    makedirs(render_path_exp, exist_ok=True)


    num = 101
    fd_list = np.linspace(0, 2 * np.pi, num)

    ap_list = np.linspace(0, 2 * np.pi, num)

    time_list = np.linspace(0, 3 * np.pi, num)
    
    # view_list = [0]
    index = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        ##only change fd
        # start = time.time()
        view.fd = 7.5*np.sin(fd_list[index]) +19.5
        view.ap = 3.2
        view.time = 0.0
        rendering_fd = render(view, gaussians, pipeline, background, fd = fd_list[index], focal_length = focal_length)["render"]       

        image_fd_tmo = torch.pow(torch.clamp((rendering_fd*2**view.time), 0.0, 1.0) + 0.0000000001, 0.5)
        torchvision.utils.save_image(image_fd_tmo, os.path.join(render_path_fd, '{0:05d}'.format(idx) + ".png"))

        ##only change ap
        view.fd = 27
        view.ap = 2.6*np.sin(ap_list[index]) + 5.0
        view.time = 0.0
        rendering_ap = render(view, gaussians, pipeline, background, focal_length = focal_length)["render"]#7.5
        image_ap_tmo = torch.pow(torch.clamp((rendering_ap*2**view.time), 0.0, 1.0) + 0.0000000001, 0.5)
        torchvision.utils.save_image(image_ap_tmo, os.path.join(render_path_ap, '{0:05d}'.format(idx) + ".png"))


        ##only change exp
        view.fd = 18
        view.ap = 1000000000000000
        view.time = 2.5*np.sin(time_list[index]) - 1.5
        rendering_exp = render(view, gaussians, pipeline, background, focal_length=focal_length)["render"]       
        image_exp_tmo = torch.pow(torch.clamp((rendering_exp*2**time_list[index]), 0.0, 1.0) + 0.0000000001, 0.5)
        torchvision.utils.save_image(image_exp_tmo, os.path.join(render_path_exp, '{0:05d}'.format(idx) + ".png"))


        rendering_exp_np = rendering_exp.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        cv2.imwrite(os.path.join(render_path_exp, '{0:05d}'.format(idx) + ".hdr"), cv2.cvtColor(rendering_exp_np, cv2.COLOR_RGB2BGR))
        index = index + 1


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, taf: bool,  focal_length):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, focal_length)

        if not skip_test:
             render_set(dataset.model_path, "test_new", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, focal_length)

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
    print('args.iteration is', args.iteration)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.taf,  args.length_focal)