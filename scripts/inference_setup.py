import copy
import json
import math
import os
import sys
from pathlib import Path

from dotdict import dotdict
import cv2
import hydra as hydra
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch


import torch.nn.functional as F


os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.abspath(os.getcwd()))

from datasets.realestate10k.realestate10k_dataset import RealEstate10kDataset
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset

from models.kyn.model import KYN
from models.kyn.model.ray_sampler import ImageRaySampler

from models.common.render import NeRFRenderer
from utils.array_operations import to, map_fn, unsqueezer
from utils.plotting import color_tensor




os.system("nvidia-smi")

gpu_id = 0

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

r, c, = 0, 0
n_rows, n_cols = 3, 3

OUT_RES = dotdict(
    X_RANGE = (-9, 9),
    Y_RANGE = (.0, .75),
    Z_RANGE = (21, 3),
    P_RES_ZX = (256, 256),
    P_RES_Y = 64
)


def plot(img, fig, axs, i=None):
    global r, c
    if r == 0 and c == 0:
        plt.show()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
    axs[r][c].imshow(img, interpolation="none")
    if i is not None:
        axs[r][c].title.set_text(f"{i}")
    c += 1
    r += c // n_cols
    c %= n_cols
    r %= n_rows
    return fig, axs


def save_plot(img, file_name=None, grey=False, mask=None, dry_run=False):
    if mask is not None:
        if mask.shape[-1] != img.shape[-1]:
            mask = np.broadcast_to(np.expand_dims(mask, -1), img.shape)
        img = np.array(img)
        img[~mask] = 0
    if dry_run:
        plt.imshow(img)
        plt.title(file_name)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))


def save_to_whole_img(img, depth, profile, file_name):
    assert img.shape[0] == depth.shape[0]
    profile = cv2.resize(profile, (img.shape[0], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_img = np.concatenate((img, depth, profile), axis=1)
    cv2.imwrite(file_name, cv2.cvtColor((combined_img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust=None):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
    if cam_incl_adjust is not None:
        xyz = xyz.view(-1, 3)
        xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        xyz_h = (cam_incl_adjust.squeeze() @ xyz_h.mT).mT
        xyz = xyz_h[:, :3].view(y_res, z_res, x_res, 3)

    return xyz


def setup_kitti360(out_folder, split="test", split_name="seg",
                    data_path=None, model_path=None, save_path=None, is_debug=False):
    resolution = (192, 640)

    data_path = "data/KITTI-360" if data_path is None else data_path
    split_path = f"datasets/kitti_360/splits/{split_name}"
    split_path = os.path.join(split_path,f"{split}_files.txt") if not is_debug else os.path.join(split_path,
                                                                                                f"{split}_files_debug.txt")
    
    dataset = Kitti360Dataset(
        data_path= data_path,
        pose_path= os.path.join(data_path, "data_poses"),
        split_path=split_path,
        return_fisheye=False,
        return_stereo=False,
        return_depth=False,
        frame_count=1,
        target_image_size=resolution,
        fisheye_rotation=(25, -25),
        color_aug=False)

    config_path = "exp_kitti_360"

    cp_path = Path(f"out/kitti_360/pretrained") if model_path is None else Path(model_path)
    
    # cp_name = cp_path.name
    # print("cp name:{}".format(cp_name))
    # cp_path = next(cp_path.glob("*.pt"))

    # NOTE enforce that the path should contain .pt filename
    assert cp_path.name[-3:] == ".pt", "the model path should .pt file"

    out_path = Path(f"media/{out_folder}/kitti_360/{cp_name}") if save_path is None else Path(save_path)

    cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  0.9961947, -0.0871557, 0],
       [0.0000000,  0.0871557,  0.9961947, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32).view(1, 4, 4)

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_kittiraw(out_folder, split="test",
                    data_path=None, model_path=None, save_path=None, is_debug=None):
    resolution = (192, 640)

    data_path = "data/KITTI-Raw" if data_path is None else data_path
    split_path = f"datasets/kitti_raw/splits/eigen_zhou/"
    split_path = os.path.join(split_path,f"{split}_files.txt") if not is_debug else os.path.join(split_path,
                                                                                                f"{split}_files_debug.txt")
    
    dataset = KittiRawDataset(
        data_path=data_path,
        pose_path="datasets/kitti_raw/orb-slam_poses",
        split_path=split_path,
        frame_count=1,
        target_image_size=resolution,
        return_stereo=True,
        return_depth=False,
        color_aug=False)

    config_path = "exp_kitti_raw"

    cp_path = Path(f"out/kitti_raw/pretrained") if model_path is None else Path(model_path)
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("*.pt"))

    out_path = Path(f"media/{out_folder}/kitti_raw/{cp_name}") if save_path is None else Path(save_path)

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_re10k(out_folder, split="test"):
    resolution = (256, 384)

    dataset = RealEstate10kDataset(
        data_path="data/RealEstate10K",
        split_path=f"datasets/realestate10k/splits/mine/{split}_files.txt" if split != "train" else None,
        frame_count=1,
        target_image_size=resolution)

    config_path = "exp_re10k"

    cp_path = Path(f"out/re10k/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("*.pt"))

    out_path = Path(f"media/{out_folder}/re10k/{cp_name}")

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def render_poses(renderer, ray_sampler, poses, projs, black_invalid=False):
    all_rays, _ = ray_sampler.sample(None, poses[:, :1], projs[:, :1])
    render_dict = renderer(all_rays, want_weights=True, want_alphas=True)

    render_dict["fine"] = dict(render_dict["coarse"])
    render_dict = ray_sampler.reconstruct(render_dict)

    depth = render_dict["coarse"]["depth"].squeeze(1)[0].cpu()
    frame = render_dict["coarse"]["rgb"][0].cpu()

    invalid = (render_dict["coarse"]["invalid"].squeeze(-1) * render_dict["coarse"]["weights"]).sum(-1).squeeze() > .8

    if black_invalid:
        depth[invalid] = depth.max()
        frame[invalid.unsqueeze(0).unsqueeze(-1), :] = 0

    return frame, depth


def render_profile(net, cam_incl_adjust, render_range_dict=None):
    # print(render_range_dict)
    if render_range_dict is None:
        q_pts = get_pts(OUT_RES.X_RANGE, OUT_RES.Y_RANGE, OUT_RES.Z_RANGE, OUT_RES.P_RES_ZX[1], OUT_RES.P_RES_Y, OUT_RES.P_RES_ZX[0], cam_incl_adjust=cam_incl_adjust)
    else:
        ppm = render_range_dict.get("ppm", None)
        if ppm is None:
            q_pts = get_pts(render_range_dict["x_range"], render_range_dict["y_range"], render_range_dict["z_range"], render_range_dict["p_res_zx"][1], render_range_dict["p_res_y"], render_range_dict["p_res_zx"][0], cam_incl_adjust=cam_incl_adjust)
        else:
            xpts = int(ppm * (render_range_dict["x_range"][1] - render_range_dict["x_range"][0]))
            zpts = int(ppm * (render_range_dict["z_range"][0] - render_range_dict["z_range"][1]))
            
            q_pts = get_pts(render_range_dict["x_range"], render_range_dict["y_range"], render_range_dict["z_range"], xpts, render_range_dict["p_res_y"], zpts, cam_incl_adjust=cam_incl_adjust)


    q_pts = q_pts.to(device).view(1, -1, 3)

    # print("q_pts:{}".format(q_pts.shape))
    batch_size = 50000
    if q_pts.shape[1] > batch_size:
        sigmas = []
        invalid = []
        l = q_pts.shape[1]
        for i in range(math.ceil(l / batch_size)):
            f = i * batch_size
            t = min((i + 1) * batch_size, l)
            q_pts_ = q_pts[:, f:t, :]
            _, invalid_, sigmas_ = net.forward(q_pts_)
            sigmas.append(sigmas_)
            invalid.append(invalid_)
        sigmas = torch.cat(sigmas, dim=1)
        invalid = torch.cat(invalid, dim=1)
    else:
        _, invalid, sigmas = net.forward(q_pts)

    sigmas[torch.any(invalid, dim=-1)] = 1
    alphas = sigmas

    if render_range_dict is None:
        alphas = alphas.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX)
    else:
        if "p_res_zx" in render_range_dict:
            alphas = alphas.reshape(render_range_dict["p_res_y"], *render_range_dict["p_res_zx"])
        else:
            alphas = alphas.reshape(render_range_dict["p_res_y"], zpts, xpts)
    
    # NOTE: shape - y_num, z_num, x_num
    # alphas_sum = torch.cumsum(alphas, dim=0)
    # print("alphas_sum shape:{}".format(alphas_sum.shape))
    # profile = (alphas_sum <= 8).float().sum(dim=0) / alphas.shape[0]

    thres = render_range_dict.get("density_acc_thres", 0.6)
    profile = (alphas < 0.6).sum(dim=0) / alphas.shape[0]

    return profile


def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def render_profile_depth(net, pred_depth, proj, pose, cam_incl_adjust, render_range_dict=None):
    # print(render_range_dict)
    if render_range_dict is None:
        q_pts = get_pts(OUT_RES.X_RANGE, OUT_RES.Y_RANGE, OUT_RES.Z_RANGE, OUT_RES.P_RES_ZX[1], OUT_RES.P_RES_Y, OUT_RES.P_RES_ZX[0], cam_incl_adjust=cam_incl_adjust)
    else:
        ppm = render_range_dict.get("ppm", None)
        if ppm is None:
            q_pts = get_pts(render_range_dict["x_range"], render_range_dict["y_range"], render_range_dict["z_range"], render_range_dict["p_res_zx"][1], render_range_dict["p_res_y"], render_range_dict["p_res_zx"][0], cam_incl_adjust=cam_incl_adjust)
        else:
            xpts = ppm * (render_range_dict["x_range"][1] - render_range_dict["x_range"][0])
            zpts = ppm * (render_range_dict["z_range"][0] - render_range_dict["z_range"][1])
            q_pts = get_pts(render_range_dict["x_range"], render_range_dict["y_range"], render_range_dict["z_range"], xpts, render_range_dict["p_res_y"], zpts, cam_incl_adjust=cam_incl_adjust)


    q_pts = q_pts.to(device).view(-1, 3)


    batch_size = 50000
    h, w = pred_depth.shape[-2:]

    if q_pts.shape[0] > batch_size:
        pred_dist = []
        invalid = []
        dists = []
        l = q_pts.shape[0]
        for i in range(math.ceil(l / batch_size)):
            f = i * batch_size
            t = min((i + 1) * batch_size, l)
            q_pts_ = q_pts[f:t, :]
            cam_pts_, dists_ = project_into_cam(q_pts_, proj, pose)
            xy_ = cam_pts_[:, :2].clamp_min(1e-3)
            # print("xy_:{}".format(xy_))
            invalid_ = (cam_pts_[:, 2:] <= 1e-3) | (xy_[:, :1] < -1) | (xy_[:, :1] > 1) | (xy_[:, 1:2] < -1) | (xy_[:, 1:2] > 1)
            # print("invalid_:{}".format(invalid_))
            invalid_ = invalid_.squeeze(-1)
            pred_dist_ = F.grid_sample(pred_depth.view(1, 1, h, w), cam_pts_[:, :2].view(1, 1, -1, 2), mode="nearest", padding_mode="border", align_corners=True).view(-1)
            # print("dist mean:{}, min:{}, max:{}".format(pred_dist_.mean(), pred_dist_.min(), pred_dist_.max()))
            # print("invalid_ after:{}".format(invalid_.shape))
            pred_dist.append(pred_dist_)
            invalid.append(invalid_)
            dists.append(dists_)

        pred_dist = torch.cat(pred_dist, dim=0)
        invalid = torch.cat(invalid, dim=0)
        dists = torch.cat(dists, dim=0)

    else:
        cam_pts, dists = project_into_cam(q_pts, proj, pose)
        xy = cam_pts[:, :2].clamp_min(1e-3)
        invalid = (cam_pts[:, 2] <= 1e-3) | (xy[:, :1] < -1) | (xy[:, :1] > 1) | (xy[:, 1:2] < -1) | (xy[:, 1:2] > 1)
        pred_dist = F.grid_sample(pred_depth.view(1, 1, h, w), cam_pts[:, :2].view(1, 1, -1, 2), mode="nearest", padding_mode="border", align_corners=True).view(-1)



    cropshape = render_range_dict.get("cropshape", False)
    # Query the density of the query points from the density field
    if cropshape:
        sigmas = ((dists >= pred_dist) & (dists <= pred_dist + 4)).float().view(-1)
    else:
        sigmas = (dists >= pred_dist).float().view(-1)


    # print("invalid:{}".format(invalid.shape))
    sigmas[invalid] = 1
    alphas = sigmas

    if render_range_dict is None:
        alphas = alphas.reshape(OUT_RES.P_RES_Y, *OUT_RES.P_RES_ZX)
    else:
        if "p_res_zx" in render_range_dict:
            alphas = alphas.reshape(render_range_dict["p_res_y"], *render_range_dict["p_res_zx"])
        else:
            alphas = alphas.reshape(render_range_dict["p_res_y"], zpts, xpts)
    

    thres = render_range_dict.get("density_acc_thres", 0.6)
    profile = (alphas < 0.6).sum(dim=0) / alphas.shape[0]

    return profile



def build_voxels(ijks, x_res, y_res, z_res, xyz, y_to_color, faces_t):
    # ijks (N, 3), N - occupied vox shape

    ids_offset = torch.tensor(
            [[1, 1, 0], [1, 0, 0],
            [0, 0, 0], [0, 1, 0],
            [1, 1, 1], [1, 0, 1],
            [0, 0, 1], [0, 1, 1]],
        dtype=torch.int32,
        device='cuda:0'
    ) # (8, 3)

    # N 8 3
    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)

    # print("ids:{}".format(ids.shape))
    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]
    # print("ids_flat:{}".format(ids_flat.shape))


    # print("xyz:{}".format(xyz.shape))
    # xyz: 3, N_sample
    # verts: 3, N_occupied * 8
    verts = xyz[:, ids_flat.reshape(-1)]
    # print("verts:{}".format(verts.shape))

    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_t.view(-1, 6, 4)

    colors = y_to_color[ijks[:, 1], :].view(-1, 1, 3).expand(-1, 6, -1)

    return verts.cpu().numpy().T, faces_off.reshape(-1, 4).cpu().numpy(), colors.reshape(-1, 3).cpu().numpy()


def get_pts_vox(x_range, y_range, z_range, x_res, y_res, z_res):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1).permute(2, 0, 1, 3)                                            # (x, y, z)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that tan(5°) = 0.0874886635
    xyz[:, :, :, 1] -= xyz[:, :, :, 2] * 0.0874886635

    return xyz


print("+++ Inference Setup Complete +++")
