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
from scipy.spatial.transform import Rotation as Rot

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, fd_path, ap_path, exp_path):
    cam_infos = []

    print('fd_path is', fd_path)
    print('ap_path is', ap_path)
    print('exp_path is', exp_path)
    fd_list = np.loadtxt(fd_path)
    time_list = np.loadtxt(exp_path)
    ap_list = np.loadtxt(ap_path)
   
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

def rotation_matrix_novel(initial_matrix, num_steps = 36):
    initial_rotation = Rot.from_matrix(initial_matrix)


    angles = np.linspace(0, 10, num=num_steps, endpoint=False) 

    rotation_axis = np.array([0, 0, 1])

    rot_matrices = []

    for angle in angles:
        rad = np.radians(angle)
        z_rotation = Rot.from_rotvec(rad * rotation_axis)
        combined_rotation = z_rotation * initial_rotation
        rot_matrices.append(combined_rotation.as_matrix())

    return rot_matrices

def translation_matrix_novel(initial_translation, num_steps = 5):

    radius = 0.4  
    center = initial_translation

    num_steps = num_steps 

    angles = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)

    translations = np.zeros((num_steps, 3))

    for i, angle in enumerate(angles):
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        translations[i] = [x, center[1], z]


    return translations

def readColmapSceneInfo(path, images, eval, fd_path, ap_path, exp_path, taf, llffhold=10):
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), fd_path=fd_path, ap_path=ap_path, exp_path=exp_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    R_new_list = rotation_matrix_novel(cam_infos[21].R, 36)
    T_new_list = translation_matrix_novel(cam_infos[21].T, 100)

    test_cam_infos_new = []
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2]

    if taf:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx == 21]
        
        new_cam = test_cam_infos[0]
        test_cam_infos_new.append(new_cam)
        for i in range(len(T_new_list)):
            new_camera_info = CameraInfo(
                    uid=21 + i, 
                    R=new_cam.R,
                    T=T_new_list[i],
                    FovY=new_cam.FovY,
                    FovX=new_cam.FovX,
                    image=new_cam.image,
                    image_path=new_cam.image_path,
                    image_name=new_cam.image_name,
                    width=new_cam.width,
                    height=new_cam.height,
                    fd = new_cam.fd, 
                    ap = new_cam.ap, 
                    time = new_cam.time 
                    )
            test_cam_infos_new.append(new_camera_info)

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

    if taf:
        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=[],
                               test_cameras=test_cam_infos_new,
                               nerf_normalization=nerf_normalization,
                               ply_path=ply_path)
    else:
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
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"]).replace('./', '')
            cam_name = os.path.join(path, frame["file_path"] + extension)
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
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fd = fd, ap = ap, time = time))
            
    return cam_infos

def novel_view_creation(cam_info, nums = 100):
    novel_views = []
    T_new_list = translation_matrix_novel_Cafe(cam_info.T, nums)
    for i in range(len(T_new_list)):
        novel_view = CameraInfo(uid = cam_info.uid + 1, 
                                R = cam_info.R,
                                T = T_new_list[i],
                                FovY=cam_info.FovY,
                                FovX=cam_info.FovX,
                                image=cam_info.image,
                                image_path=cam_info.image_path,
                                image_name=cam_info.image_name,
                                width=cam_info.width,
                                height=cam_info.height,
                                fd = cam_info.fd, 
                                ap = cam_info.ap, 
                                time = cam_info.time 
                                )
        novel_views.append(novel_view)
    return novel_views

def readNerfSyntheticInfo(path, white_background, eval, taf, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    if taf:
        test_cam_infos_new = novel_view_creation(train_cam_infos[34], 1)
    else:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if taf:
        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=train_cam_infos,
                               test_cameras=test_cam_infos_new,
                               nerf_normalization=nerf_normalization,
                               ply_path=ply_path)
    else:
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