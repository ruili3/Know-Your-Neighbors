import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.kyn.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z
from PIL import Image
import numpy as np
from models.kyn.model.models_kyn import KYN


IDX = 0
EPS = 1e-4

# The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  0.9961947,  0.0871557, 0],
       [0.0000000, -0.0871557,  0.9961947, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32
).view(1, 1, 4, 4)


def get_pts(x_range, y_range, z_range, ppm, ppm_y, y_res=None, specify_yslice=None):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    if y_res == 1:
        if specify_yslice is None:
            y = torch.tensor([y_range[0] * .5 + y_range[1] * .5]).view(y_res, 1, 1).expand(-1, z_res, x_res)
        else:
            y = torch.tensor([specify_yslice]).view(y_res, 1, 1).expand(-1, z_res, x_res)
    else:
        y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


# This function takes all points between min_y and max_y and projects them into the x-z plane.
# To avoid cases where there are no points at the top end, we consider also points that are beyond the maximum z distance.
# The points are then converted to polar coordinates and sorted by angle.

def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    slices = []
    # y-axis sampled positions
    ys = torch.linspace(y_range[0], y_range[1], y_res)
    # print("y_res:{}".format(y_res))
    if y_res > 1:
        slice_height = ys[1] - ys[0]
    else:
        slice_height = 0
    n_bins = 360

    for y in ys:
        if y_res == 1:
            min_y = y
            max_y = y_range[-1]
        else:
            min_y = y - slice_height / 2
            max_y = y + slice_height / 2

        slice = []

        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T

            # print("pc_world shape:{}".format(pc_world.shape))
            mask = ((pc_world[:, 1] >= min_y) & (pc_world[:, 1] <= max_y)) | (torch.norm(pc_world[:, :3], dim=-1) >= max_dist)

            # NOTE get the x/y axis coordinate? why use the first two?
            slice_points = pc[mask, :2]

            # NOTE arctan of y/x: the angle of the polar coordinates
            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            # NOTE the distance of the polar coordinates
            dists = torch.norm(slice_points, dim=-1)

            slice_points_polar = torch.stack((angles, dists), dim=1)
            # print("slice_points_polar:{}".format(slice_points_polar.shape))

            # Sort by angles for fast lookup
            slice_points_polar = slice_points_polar[torch.sort(angles)[1], :] # NOTE torch.sort(angles)[1]: indices

            slice_points_polar_binned = torch.zeros_like(slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(-math.pi, math.pi, n_bins+1, device=slice_points_polar.device)

            dist = slice_points_polar[0, 1]

            # To reduce noise, we bin the lidar points into bins of 1deg and then take the minimum distance per bin.
            # NOTE torch.searchsorted --  The torch.searchsorted function in PyTorch is used to find the [indices] at which elements should be inserted to maintain the order of an array. It's similar to NumPy's searchsorted function.
            border_is = torch.searchsorted(slice_points_polar[:, 0], bin_borders)

            for i in range(n_bins):
                left_i, right_i = border_is[i], border_is[i+1]
                angle = (bin_borders[i] + bin_borders[i+1]) * .5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])
                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned

            # Append first element to last to have full 360deg coverage
            slice_points_polar = torch.cat(( torch.tensor([[slice_points_polar[-1, 0] - math.pi * 2, slice_points_polar[-1, 1]]], device=slice_points_polar.device), slice_points_polar, torch.tensor([[slice_points_polar[0, 0] + math.pi * 2, slice_points_polar[0, 1]]], device=slice_points_polar.device)), dim=0)

            slice.append(slice_points_polar)

            # print("360 slice shape:{}".format(slice_points_polar.shape))

        slices.append(slice)

    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)

    thresh = (len(slices[0]) - 2) / len(slices[0])
    # print("thres:{}".format(thresh))

    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)

    world_to_velos = torch.inverse(velo_poses)

    step = pts.shape[0] // len(slices)
    
    # [slices] is a list with len of 1
    for i, slice in enumerate(slices):
        # [slice] is a list with size of 20: 20 lidar scans
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice, world_to_velos)):
            
            # print("lidar polar:{}".format(lidar_polar.shape))
            # print("world_to_velo:{}".format(world_to_velo.shape))
            # print("j:{}".format(j))

            # camera -> velo
            pts_velo = (world_to_velo @ pts[i*step: (i+1)*step, :].T).T

            # Convert query points to polar coordinates in velo space
            angles = torch.atan2(pts_velo[:, 1], pts_velo[:, 0])
            dists = torch.norm(pts_velo, dim=-1)

            indices = torch.searchsorted(lidar_polar[:, 0].contiguous(), angles)

            left_angles = lidar_polar[indices-1, 0]
            right_angles = lidar_polar[indices, 0]

            left_dists = lidar_polar[indices-1, 1]
            right_dists = lidar_polar[indices, 1]

            interp = (angles - left_angles) / (right_angles - left_angles)
            surface_dist = left_dists * (1 - interp) + right_dists * interp

            is_occupied_velo = (dists > surface_dist) | (dists < min_dist)

            is_occupied[i*step: (i+1)*step] += is_occupied_velo.float()

            if j == 0:
                is_visible[i*step: (i+1)*step] |= ~is_occupied_velo

    # len(slices[0]): 20
    is_occupied /= len(slices[0])

    is_occupied = is_occupied > thresh

    return is_occupied, is_visible


def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def plot(pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()

    rows = math.ceil(yd / 2)
    fig, axs = plt.subplots(rows, 2)

    for y in range(yd):
        r = y // 2
        c = y % 2

        if rows > 1:
            axs[r][c].imshow(pts[y], interpolation="none")
        else:
            axs[c].imshow(pts[y], interpolation="none")
    plt.show()


def plot_sperical(polar_pts):
    polar_pts = polar_pts.cpu()
    angles = polar_pts[:, 0]
    dists = polar_pts[:, 1]

    max_dist = dists.mean() * 2
    dists = dists.clamp(0, max_dist) / max_dist

    x = -torch.sin(angles) * dists
    y = torch.cos(angles) * dists

    plt.plot(x, y)
    plt.show()


def save(name, pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()[0]
    plt.imsave(name, pts)


def save_all(f, is_occupied, is_occupied_pred, images, xd, yd, zd):
    save(f"{f}_gt.png", is_occupied, xd, yd, zd)
    save(f"{f}_pred.png", is_occupied_pred, xd, yd, zd)
    plt.imsave(f"{f}_input.png", images[0, 0].permute(1, 2, 0).cpu().numpy() * .5 + .5)


def posenc(d_hid, n_samples):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.from_numpy(sinusoid_table).unsqueeze(0)
    return sinusoid_table


class KYNWrapper(nn.Module):
    def __init__(self, renderer, config, dataset) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        self.x_range = config.get("x_range", (-4, 4))
        self.y_range = config.get("y_range", (0, .75)) 
        self.z_range = config.get("z_range", (20, 4))
        self.ppm =  config.get("ppm", 10)
        self.ppm_y = config.get("ppm_y", 4)
        self.y_res = config.get("y_res", 1)
        self.specify_yslice = config.get("specify_yslice", None)

        # If some region is behind the farest visible surface, cut this region off since there is no visible cues to infer geometry for this area.
        self.cut_far_invisible_area = config.get("cut_far_invisible_area", False)
        self.save_gt_occ_map_path = config.get("save_gt_occ_map_path", "")
        self.read_gt_occ_path = config.get("read_gt_occ_path", "")

        # Objects-level evaluation settings
        self.is_eval_object = config.get("is_eval_object", False)
        self.read_gt_obj_path = config.get("read_gt_obj_path", None)
        self.obj_z_expand = int(config.get("obj_z_expand", 0) * self.ppm)
        self.obj_x_expand = int(config.get("obj_x_expand", 0) * self.ppm)

        if self.is_eval_object:
            self.obj_acc = []
            self.obj_prec = []
            self.obj_rec = []

            self.obj_ie_acc = []
            self.obj_ie_prec = []
            self.obj_ie_rec = []



        self.count = 0
        self.data_len = dataset.get_length
        print("self datalen:{}".format(self.data_len))
   

        print("x_range:{}, y_range:{}, z_range:{}, ppm:{}, ppm_y:{}, y_res:{}".format(self.x_range,
                                                                            self.y_range, self.z_range,
                                                                            self.ppm, self.ppm_y, self.y_res))

        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)

        self.dataset = dataset
        self.aggregate_timesteps = config.get("gt_aggregate_timesteps", 20)

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]


    def compute_object_occ_scores(self, obj_locs_map, is_occupied, is_occupied_pred, is_visible):
        '''
        obj_locs_map: 0 - not_occupied, 255 - occupied, 1,2,3,... object ids
        '''
        obj_ids = torch.unique(obj_locs_map)[1:-1]

        h, w = obj_locs_map.shape

        # get valid map where only object areas are 1
        obj_valid_map = is_occupied.new_zeros(is_occupied.shape)

        # no annotated objects (objcts id starts from 1)
        if (obj_locs_map != 0).sum() == 0:
            nan_val = torch.tensor(float('nan'), device=obj_locs_map.device)
            return (nan_val.clone(), nan_val.clone(), nan_val.clone())

        for ii in range(obj_ids.shape[0]):
            obj_id = obj_ids[ii]
            mask = (obj_locs_map == obj_id).nonzero()
            y_min, x_min = mask.min(0).values
            y_max, x_max = mask.max(0).values

            # add area expansions
            y_min = max(y_min-self.obj_z_expand, 0)
            y_max = min(y_max+self.obj_z_expand, h)

            x_min = max(x_min-self.obj_x_expand, 0)
            x_max = min(x_max+self.obj_x_expand, w)

            obj_valid_map[y_min:y_max, x_min:x_max] = 1


        obj_is_occupied = is_occupied[obj_valid_map]
        obj_is_occupied_pred = is_occupied_pred[obj_valid_map]
        obj_is_visible = is_visible[obj_valid_map]

        is_occupied = None
        is_occupied_pred = None
        is_visible = None

        obj_o_acc = (obj_is_occupied_pred == obj_is_occupied).float().mean().item()
        obj_ie_acc = (obj_is_occupied_pred == obj_is_occupied)[(~obj_is_visible)].float().mean().item()
        obj_ie_rec = (~obj_is_occupied_pred)[(~obj_is_occupied) & (~obj_is_visible)].float().mean().item()

        return (obj_o_acc, obj_ie_acc, obj_ie_rec)
        


    def compute_scene_occ_scores(self, is_occupied, is_occupied_pred, is_visible):
        scene_o_acc = (is_occupied_pred == is_occupied).float().mean().item()
        scene_ie_acc = (is_occupied_pred == is_occupied)[(~is_visible)].float().mean().item()
        scene_ie_rec = (~is_occupied_pred)[(~is_occupied) & (~is_visible)].float().mean().item()

        return (scene_o_acc, scene_ie_acc, scene_ie_rec)

    def mean_ignore_nan(self, lst):
        filtered_lst = [x for x in lst if not math.isnan(x)]
        if len(filtered_lst) == 0:
            return float('nan')  # Return nan if all values are nan or the list is empty
        return sum(filtered_lst) / len(filtered_lst)


    def sepcify_croping_bound(self, defined_z_min=4.0, defined_z_max=50.0):
        curr_z_min = self.z_range[1]
        curr_z_max = self.z_range[0]
        assert curr_z_min >= defined_z_min and curr_z_max <= defined_z_max
        upper_pixel_margin = int(self.ppm * (defined_z_max - curr_z_max))
        lower_pixel_margin = int(self.ppm * (curr_z_min - defined_z_min))

        return upper_pixel_margin, lower_pixel_margin


    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                 # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        index = data["index"].item()

        self.count += 1

        # NOTE: `_img_ids` stores the non-continuous images used in kitti360. To accumulate the lidar points,
        # one needs to maintain a continuous index list whose value indicates the position of `_img_ids`. This
        # index list is `datapoints`

        # [datapoints] `295` indicates the position of `_img_ids`, _img_ids[295] is the real image index.png
        # datapoints:[('2013_05_28_drive_0000_sync', 295, False), ('2013_05_28_drive_0000_sync', 315, False), ('2013_05_28_drive_0000_sync', 335, False), ('2013_05_28_drive_0000_sync', 355, False)]
        
        # [_img_ids]: real images ids
        # _img_ids:{'2013_05_28_drive_0009_sync': array([    1,    80,    83, ..., 13941, 13943, 13955]), '2013_05_28_drive_0010_sync': array([   1,   10,   13, ..., 3657, 3684, 3743]), '2013_05_28_drive_0007_sync': array([   1,    2,    3, ..., 3148, 3151, 3161]), '2013_05_28_drive_0000_sync': array([    1,     9,    14, ..., 11496, 11498, 11501]), '2013_05_28_drive_0005_sync': array([   1,   30,   33, ..., 6715, 6718, 6723]), '2013_05_28_drive_0003_sync': array([   1,    4,    6, ..., 1028, 1029, 1030]), '2013_05_28_drive_0002_sync': array([    1,    43,    51, ..., 19227, 19229, 19231]), '2013_05_28_drive_0006_sync': array([   1,   35,   42, ..., 9696, 9697, 9698]), '2013_05_28_drive_0004_sync': array([    1,    54,    57, ..., 11398, 11399, 11400])}

        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]

        init_id = id

        n, v, c, h, w = images.shape
        device = images.device

        T_velo_to_pose = torch.tensor(self.dataset._calibs["T_velo_to_pose"], device=device)

        # Our coordinate system is at the same position as cam0, but rotated 5deg up along the x axis to adjust for camera inclination. Consequently, the xz plane is parallel to the street.
        world_transform = torch.inverse(poses[:, :1, :, :]) # transform to camera0
        world_transform = cam_incl_adjust.to(device) @ world_transform # add inclination
        poses = world_transform @ poses

        self.sampler.height = h
        self.sampler.width = w

        # Load lidar pointclouds
        points_all = []
        velo_poses = []

        
        # if no GT occupancy map, read it from lidar points, else read from npy file.
        if self.read_gt_occ_path == "":
            # NOTE: points_all contains xxxxx frame point clouds
            for id in range(id, min(id + self.aggregate_timesteps, seq_len)):
                points = np.fromfile(os.path.join(self.dataset.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{self.dataset._img_ids[seq][id]:010d}.bin"), dtype=np.float32).reshape(-1, 4)
                points[:, 3] = 1.0
                points = torch.tensor(points, device=device)
                # NOTE also transfer velo coods to cam0
                velo_pose = world_transform.squeeze() @ torch.tensor(self.dataset._poses[seq][id], device=device) @ T_velo_to_pose
                points_all.append(points)
                velo_poses.append(velo_pose)

            velo_poses = torch.stack(velo_poses, dim=0)
        else:
            name = "{}_{:010d}".format(seq, self.dataset._img_ids[seq][init_id])
            occ_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_occgt.npy"))
            vis_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_visgt.npy"))
            is_occupied = torch.from_numpy(occ_gt).bool().to(device)
            is_visible = torch.from_numpy(vis_gt).bool().to(device)

            z_size = self.ppm * (self.z_range[0]-self.z_range[1])
            curr_gt_size = is_occupied.shape[0]
            # when evaluation max_distance is smaller than gt max_distance, align the prediction and gt occ map
            if z_size < curr_gt_size:
                upper_pixel_margin, lower_pixel_margin = self.sepcify_croping_bound(defined_z_min=4.0,
                                                                                    defined_z_max=50.0)
                is_occupied = is_occupied[upper_pixel_margin:curr_gt_size-lower_pixel_margin,:]
                is_visible = is_visible[upper_pixel_margin:curr_gt_size-lower_pixel_margin,:]


        rays, _ = self.sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])

        ids_encoder = [0]
        self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
        self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
        self.renderer.net.set_scale(0)

        # The points are sampled from the cuboid area ranging from the lower-uppper bound of x/y/z axis
        q_pts, (xd, yd, zd) = get_pts(self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y, self.y_res, self.specify_yslice)
        q_pts = q_pts.to(images.device).view(-1, 3)


        # Query the density of the query points from the density field
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            res = self.renderer.net(q_pts_.unsqueeze(0), only_density=True)
            densities_ = res[2]
            densities.append(densities_.squeeze(0))
        densities = torch.cat(densities, dim=0).squeeze()
        is_occupied_pred = densities > self.occ_threshold

        # 'self.read_gt_occ_path' specifies the path of pre-computed GT occupany. If not specified, generate GT Occ from LIDAR points (a bit time consuming for 300-frame accumulated gt).
        if self.read_gt_occ_path == "":
            slices = get_lidar_slices(points_all, velo_poses, self.y_range, yd, (self.z_range[0] ** 2 + self.x_range[0] ** 2) ** .5)
            is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)        

        if self.save_gt_occ_map_path != "":
            name = "{}_{:010d}".format(seq, self.dataset._img_ids[seq][init_id])
            print("save GT of {}".format(name))
            is_occupied_save = is_occupied.reshape(zd, xd).cpu().numpy()
            is_visible_save = is_visible.reshape(zd, xd).cpu().numpy()
            np.save(os.path.join(self.save_gt_occ_map_path, name + "_occgt.npy"), is_occupied_save)
            np.save(os.path.join(self.save_gt_occ_map_path, name + "_visgt.npy"), is_visible_save)


        is_occupied = is_occupied.reshape(zd, xd)
        is_occupied_pred = is_occupied_pred.reshape(zd, xd)
        is_visible = is_visible.reshape(zd, xd)
        is_occupied &= ~is_visible

        # evaluate the metrics for object area.
        if self.is_eval_object:
            assert self.read_gt_obj_path != "" and self.read_gt_obj_path is not None
            name = "{}_{:010d}".format(seq, self.dataset._img_ids[seq][init_id])
            obj_locs_map = Image.open(os.path.join(self.read_gt_obj_path, name + "_occgt_anno.png"))
            obj_locs_map = torch.from_numpy(np.array(obj_locs_map).astype(np.uint8)).to(device)

            z_size = self.ppm * (self.z_range[0]-self.z_range[1])
            curr_gt_size = obj_locs_map.shape[0]
            if z_size < curr_gt_size:
                obj_locs_map = obj_locs_map[upper_pixel_margin:curr_gt_size-lower_pixel_margin,:]
            # object-level metrics
            obj_eval_res = self.compute_object_occ_scores(obj_locs_map, is_occupied, is_occupied_pred, is_visible)
        else:
            obj_eval_res = (0, 0, 0)

        
        if self.cut_far_invisible_area:
            z_indices = torch.nonzero(is_visible)[:, 0]
            if z_indices.shape[0] == 0:
                print("no visible area in the testing image, skip")
                data["scene_O_acc"] = torch.tensor(float('nan'), device=images.device)
                data["scene_IE_acc"] = torch.tensor(float('nan'), device=images.device)
                data["scene_IE_rec"] = torch.tensor(float('nan'), device=images.device)
                data["object_O_acc"] = torch.tensor(float('nan'), device=images.device)
                data["object_IE_acc"] = torch.tensor(float('nan'), device=images.device)
                data["object_IE_rec"] = torch.tensor(float('nan'), device=images.device)
                globals()["IDX"] += 1
                return data
            # Cut off the area that is behind the farest visible surface of the camera
            z_min_val = torch.min(z_indices)
            z_min_val = max(0, z_min_val - 2 * self.ppm) # cut off areas 2m behind the surface 
            h, w  = is_occupied.shape
            is_occupied = is_occupied[z_min_val:h, :]
            is_visible = is_visible[z_min_val:h, :]
            is_occupied_pred = is_occupied_pred[z_min_val:h, :]

        # scene-level metrics
        scene_eval_res = self.compute_scene_occ_scores(is_occupied, is_occupied_pred, is_visible)

        data["scene_O_acc"] = scene_eval_res[0]
        data["scene_IE_acc"] = scene_eval_res[1]
        data["scene_IE_rec"] = scene_eval_res[2]
        data["object_O_acc"] = obj_eval_res[0]
        data["object_IE_acc"] = obj_eval_res[1]
        data["object_IE_rec"] = obj_eval_res[2]

        globals()["IDX"] += 1

        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False, drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["scene_O_acc", "scene_IE_acc", "scene_IE_rec", "object_O_acc", "object_IE_acc", "object_IE_rec"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "KYN")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = KYNWrapper(
        renderer,
        config["model_conf"],
        make_test_dataset(config["data"])
    )

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass