import os
import os
import sys
from pathlib import Path
import cv2
import hydra as hydra
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import open_dict
from torch import nn
from tqdm import tqdm
sys.path.append(os.path.abspath(os.getcwd()))
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from utils.array_operations import map_fn, unsqueezer
from torch.utils.data import DataLoader
from models.kyn.model import KYN
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from models.common.render import NeRFRenderer
import math

os.system("nvidia-smi")
resolution = (192, 640)
gpu_id = 1

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True



ids_offset = torch.tensor(
        [[1, 1, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [1, 1, 1], [1, 0, 1],
        [0, 0, 1], [0, 1, 1]],
    dtype=torch.int32,
    device=device)

def build_voxel(i, j, k, x_res, y_res, z_res, xyz, offset):
    ids = [[i+1, j+1, k], [i+1, j, k],
           [i, j, k], [i, j+1, k],
           [i+1, j+1, k+1], [i+1, j, k+1],
           [i, j, k+1], [i, j+1, k+1]]

    faces_off = [[v+offset for v in f] for f in faces]

    ids_flat = list(map(lambda ijk: ijk[0]*y_res*z_res + ijk[1]*z_res + ijk[2], ids))

    verts = xyz[:, ids_flat].cpu().numpy().T

    colors = np.tile(np.array(plt.cm.get_cmap("magma")(1 - (verts[..., 1].mean().item() - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))[:3]).reshape((1, 3)), ((len(faces_off), 1)))
    colors = (colors * 255).astype(np.uint8)

    return verts, faces_off, colors




def build_voxels(ijks, x_res, y_res, z_res, xyz, y_to_color, faces_t):
    # ijks (N, 3), N - occupied vox shape

    # N 8 3
    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)

    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]

    verts = xyz[:, ids_flat.reshape(-1)]

    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_t.view(-1, 6, 4)

    colors = y_to_color[ijks[:, 1], :].view(-1, 1, 3).expand(-1, 6, -1)

    return verts.cpu().numpy().T, faces_off.reshape(-1, 4).cpu().numpy(), colors.reshape(-1, 3).cpu().numpy()


def get_pts(x_range, y_range, z_range, x_res, y_res, z_res):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1).permute(2, 0, 1, 3)                                            # (x, y, z)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that tan(5°) = 0.0874886635
    xyz[:, :, :, 1] -= xyz[:, :, :, 2] * 0.0874886635

    return xyz

def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist



@hydra.main(version_base=None, config_path="../configs/", config_name = "gen_voxel.yaml")
def main(config):
    print('Loading dataset')

    resolution = (192, 640)

    model_type = config["vis_setting"]["model_type"]
    assert model_type in ["kitti_raw", "kitti_360", "real10k"]

    out_path = Path(config["vis_setting"]["save_path"])

    os.makedirs(out_path, exist_ok=True)

    model_path = config["vis_setting"]["model_path"]
    vis_offset = config["vis_setting"]["vis_offset"]
    is_debug = config["vis_setting"]["is_debug"]
    data_path = config["vis_setting"]["data_path_main"] # if is_debug else os.path.join(tmpdir, "KITTI-360")   



    render_range_dict = config["vis_setting"]["render_range_dict"]
    X_RANGE = render_range_dict["x_range"]
    Y_RANGE = render_range_dict["y_range"]
    Z_RANGE = render_range_dict["z_range"]


    ppm = render_range_dict["ppm"]
    p_res_x = int(ppm * abs(X_RANGE[0]-X_RANGE[1]))
    p_res_z = int(ppm * abs(Z_RANGE[0]-Z_RANGE[1]))
    p_res_y = render_range_dict["p_res_y"]
    p_res = [p_res_z, p_res_x]

    picklist_path = render_range_dict.get("pick_list_path", "")
    if picklist_path != "":
        with open(picklist_path, 'r') as ff:
            lines = ff.readlines()
        picklist = [int(line.strip()) for line in lines]
    else:
        picklist = None


    faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
    faces_t = torch.tensor(faces, device=device)

    y_steps = (1 - (torch.linspace(0, 1 - 1/p_res_y, p_res_y) + 1 / (2 * p_res_y))).tolist()
    cmap = plt.cm.get_cmap("magma")
    y_to_color = (torch.tensor(list(map(cmap, y_steps)), device=device)[:, :3] * 255).to(torch.uint8)


    print(f"X_RANGE:{X_RANGE}, Y_RANGE:{Y_RANGE}, Z_RANGE:{Z_RANGE}, p_res:{p_res}, p_res_y:{p_res_y}")

    split_path = f"datasets/kitti_360/splits/seg"
    split_path = os.path.join(split_path,f"test_files.txt") if not is_debug else os.path.join(split_path, f"test_files_debug.txt")


    cp_path = Path(f"out/kitti_360/pretrained") if model_path is None else Path(model_path)
    assert cp_path.name[-3:] == ".pt", "the model path should .pt file"

    dataset = Kitti360Dataset(
        data_path=data_path,
        pose_path=os.path.join(data_path, "data_poses"),
        split_path=split_path,
        return_fisheye=False,
        return_stereo=False,
        return_depth=False,
        frame_count=1,
        target_image_size=resolution,
        fisheye_rotation=(25, -25),
        color_aug=False)

    print('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    with open_dict(config):
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "distance"
        config["model_conf"]["grid_learn_empty"] = False


    net = globals()[config["model_conf"]["arch"]](config["model_conf"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 256
    renderer.renderer.lindisp = False

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    if "model" in cp:
        _wrapper.load_state_dict(cp["model"], strict=False)
    else:
        _wrapper.load_state_dict(cp, strict=False)
    renderer.to(device)
    renderer.eval()

    with torch.no_grad():


        if picklist is not None:
            iter_list = picklist
        else:
            iter_list = range(0, len(dataset), vis_offset)

        print("iterlist:{}".format(iter_list))

        for idx in iter_list:
            data = dataset[idx]

            seq, init_id, is_right = dataset._datapoints[idx]
            img_id = int(dataset._img_ids[seq][init_id])

            print("generating {:010d}_{}_{:010d}".format(idx, seq, img_id))

            data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)

            images = torch.stack(data_batch["imgs"], dim=1).to(device)
            poses = torch.stack(data_batch["poses"], dim=1).to(device)
            projs = torch.stack(data_batch["projs"], dim=1).to(device)

            poses = torch.inverse(poses[:, :1]) @ poses

            n, nv, c, h, w = images.shape

            # use nerf-based methods
            q_pts = get_pts(X_RANGE, Y_RANGE, Z_RANGE, p_res[1], p_res_y, p_res[0])
            q_pts = q_pts.to(device).reshape(1, -1, 3)

            net.compute_grid_transforms(projs, poses)
            net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])
            net.set_scale(0)    
            _, invalid, sigmas = net.forward(q_pts)


            alphas = sigmas
            alphas = alphas.reshape(1, 1, p_res[1], p_res_y, p_res[0])              # (x, y, z)
            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)            
            is_occupied = alphas_mean.squeeze() > .5


            verts, faces, colors = build_voxels(is_occupied.nonzero(), p_res[1], p_res_y, p_res[0], q_pts.squeeze(0).T, y_to_color, faces_t)

            verts = list(map(tuple, verts))
            verts_data = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

            face_data = np.array(faces, dtype='i4')
            color_data = np.array(colors, dtype='u1')
            ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,)),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

            ply_faces['vertex_indices'] = face_data
            ply_faces["red"] = color_data[:, 0]
            ply_faces["green"] = color_data[:, 1]
            ply_faces["blue"] = color_data[:, 2]

            verts_el = PlyElement.describe(verts_data, "vertex")
            faces_el = PlyElement.describe(ply_faces, "face")

            save_name = "{:010d}_{}_{:010d}_z{}_{}_x{}_{}_ppm{}_y{}".format(idx, seq, img_id, 
                                                                                *Z_RANGE, *X_RANGE, ppm, p_res_y)

            PlyData([verts_el, faces_el]).write(str(out_path / f"{save_name}.ply"))

    pass


if __name__ == '__main__':
    main()
