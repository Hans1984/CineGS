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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fd: float
    ap: float
    time: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
#     #fd_list = [35, 19, 21, 20, 22, 23, 21, 20, 35, 34, 33, 36, 35, 35, 35, 35, 35, 35, 36, 34, 36, 57, 58, 58, 33, 58, 60, 61, 57, 40, 56, 58, 33, 60, 61, 61, 61, 62, 61, 64, 48, 63, 63]
#     #ap_list = [22, 5, 5, 5, 5, 5, 5, 5, 22, 5, 5, 5, 5, 5, 5, 5, 22, 5, 5, 5, 5, 5, 5, 5, 22, 5, 5, 5, 5, 5, 5, 5, 22, 5, 5, 5, 5, 5, 5, 5, 22, 5, 5]

#     fd_list = [82, 24.125, 23.625, 23.875, 24.625, 24.25, 24.625, 48, 22.875, 23.625, 23, 23.125, 41.75, 41.5, 37.5, 41.5, 41.75, 40.25, 49.25, 53.75, 54, 39.25, 41.5, 43, 60.25, 63, 62.5, 97, 41.25, 460, 420, 420, 420, 400, 432, 41.25, 63, 119, 100, 98, 98]#[23.7, 17.8, 17.2, 16, 16.12, 16.12, 27.37, 16.12, 15.62, 14.37, 24.87, 26.75, 25.62, 25.5, 24.62, 26.25, 25, 26.75, 27.25, 26, 25, 25.375, 24, 25.625, 28.12, 28, 33, 32.75, 34.75, 33.25, 24.25, 31.75, 67, 67.5, 57.5, 52.25, 29, 47.25, 51, 51.75, 48.75, 49.75, 29.625, 51.25]
#     # fd_list = np.linspace(2.9, 18, 194)
#     ap_list = [20, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4]
# #[22, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 4, 22, 4]
#     time_list = []
    ##TAF_54
    # fd_list_orig = np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/fd_list_np_exif_sparse_54')#[7.0, 9.0, 11.0, 6.9, 8.7, 11.0, 6.9, 8.5, 10.9, 6.8, 8.7, 10.9, 7.0, 8.5, 10.8, 7.3, 8.4, 10.8, 7.1, 8.1, 10.4, 7.1, 8.0, 10.2, 7.3, 7.9, 10.1, 7.06, 7.93, 10.25, 7.1, 8.3, 10.3, 7.18, 8.18, 10.25, 7.45, 8.69, 10.25, 7.38, 8.5, 10.06, 7.47, 8.44, 10.62, 7.96, 9.125, 12.06, 8.25, 9.25, 11.93, 7.81, 9.25, 12.06]#np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/fd_list_np_lstsq')#
    # fd_list = [x * 8.687 for x in fd_list_orig]
    # time_list_orig = [2.0, 0.0, -2.0704]#[-1.0, 0.0, 1.0]#[1/250.0, 1/60.0, 1/15.0]
    # time_list = time_list_orig * 18
    # # time_list = [-2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, -2.0, 0.0, -2.0, 2.0, 2.0, -2.0, 0.0, 2.0, 2.0, -2.0, 0.0]
    # ap_list = [4] * 54

    fd_list_orig = np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/exif_info/fd_list_np_exif_sparse_ana3')#np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/exif_info/fd_list_np_exif_sparse_81')#[7.0, 9.0, 11.0, 6.9, 8.7, 11.0, 6.9, 8.5, 10.9, 6.8, 8.7, 10.9, 7.0, 8.5, 10.8, 7.3, 8.4, 10.8, 7.1, 8.1, 10.4, 7.1, 8.0, 10.2, 7.3, 7.9, 10.1, 7.06, 7.93, 10.25, 7.1, 8.3, 10.3, 7.18, 8.18, 10.25, 7.45, 8.69, 10.25, 7.38, 8.5, 10.06, 7.47, 8.44, 10.62, 7.96, 9.125, 12.06, 8.25, 9.25, 11.93, 7.81, 9.25, 12.06]#np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/fd_list_np_lstsq')#
    fd_list =  [x * 13.94 for x in fd_list_orig]#[x * 24.26 for x in fd_list_orig]#[x * 21.34 for x in fd_list_orig]
    time_list = np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/exif_info/exp_list_np_blur_sparse_ana3')#np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/exif_info/exp_list_np_blur_sparse_81')#[2.0, 0.0, -2.0704]#[-1.0, 0.0, 1.0]#[1/250.0, 1/60.0, 1/15.0]
    ap_list = np.loadtxt('/HPS/InfNerf/work/Chao/Gaussian_splatting_conv_2d_dof_hdr_python/exif_info/ap_list_np_exif_sparse_ana3')#[4] * 54

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        # print('height and width is', height, width)
        # exit()
        fd = fd_list[idx]
        ap = ap_list[idx]
        time = time_list[idx]
        # print('idx is', idx)

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fd = fd, ap = ap, time = time )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=10):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        # print('fovx is', fovx)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"]).replace('./', '')
            #cam_name = os.path.join(path, frame["file_path"] + extension)
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            fd = float(np.array(frame["focus_distance"]))
            ap = float(np.array(frame["aperture"]))
            time = float(np.array(frame["exp"]))
            # print('fd is', fd)
            # print('ap is', ap)
            # print('time is', time)
            # exit()
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fd = fd, ap = ap, time = time))
            
    return cam_infos


def generate_random_points(num_pts, bounds):
    """
    Generate random points within a specified bounding box.
    Args:
    num_pts (int): Number of points to generate.
    bounds (tuple): Boundaries for the random points as (min_x, min_y, min_z, max_x, max_y, max_z).
    Returns:
    np.ndarray: Array of random points with shape (num_pts, 3).
    """
    # 解构 bounds 元组
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    
    # 生成每个维度的随机数并调整范围
    x = min_x + (max_x - min_x) * np.random.random(num_pts)
    y = min_y + (max_y - min_y) * np.random.random(num_pts)
    z = min_z + (max_z - min_z) * np.random.random(num_pts)

    # 合并各维度的随机数为一个 (num_pts, 3) 形状的数组
    random_points = np.column_stack((x, y, z))

    return random_points


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        print(f"Generating random point cloud ({num_pts})...")
        bounds = (-17.0, -22.0, -13.0, 7.78, 7.05, 12.449)#(-4.5, -9.5, 0, 4.26, 3.5, 4.84)#
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = generate_random_points(num_pts, bounds)
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) *4.0 - 2.0
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}