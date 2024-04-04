import sys
from argparse import ArgumentParser
sys.path.append(".")
from scripts.inference_setup import *
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
from models.kyn.model import KYN, ImageRaySampler
from models.common.render import NeRFRenderer
from utils.array_operations import map_fn, unsqueezer
from utils.plotting import color_tensor
import os
from plyfile import PlyData, PlyElement

def parse_ply(verts, faces, colors):


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

    return verts_el, faces_el


def main():
    parser = ArgumentParser("Generate density field from single image.")
    parser.add_argument("--img", required=True, help="Path to the image.")
    parser.add_argument("--model_path", required=True, help="Path to the pre-trained model.")
    parser.add_argument("--save_path", required=True, help="Path to save the predictions.")
    args = parser.parse_args()
    resolution = (192, 640)
    config_path = "demo"

    cam_incl_adjust = torch.tensor(
        [[1.0000000, 0.0000000, 0.0000000, 0],
            [0.0000000, 0.9961947, -0.0871557, 0],
            [0.0000000, 0.0871557, 0.9961947, 0],
            [0.0000000, 000000000, 0.0000000, 1]
            ],
        dtype=torch.float32).view(1, 4, 4)

    proj = torch.tensor([
        [ 0.7849,  0.0000, -0.0312, 0],
        [ 0.0000,  2.9391,  0.2701, 0],
        [ 0.0000,  0.0000,  1.0000, 0],
        [ 0.0000,  0.0000,  0.0000, 1],
    ], dtype=torch.float32).view(1, 4, 4)



    initialize(version_base=None, config_path="../configs", job_name="demo")
    config = compose(config_name=config_path, overrides=[])

    render_range_dict = config["render_range_dict"]
    X_RANGE = render_range_dict["x_range"]
    Y_RANGE = render_range_dict["y_range"]
    Z_RANGE = render_range_dict["z_range"]
    ppm = render_range_dict["ppm"]
    p_res_x = int(ppm * abs(X_RANGE[0]-X_RANGE[1]))
    p_res_z = int(ppm * abs(Z_RANGE[0]-Z_RANGE[1]))
    p_res_y = render_range_dict["p_res_y"]
    p_res = [p_res_z, p_res_x]

    y_steps = (1 - (torch.linspace(0, 1 - 1/p_res_y, p_res_y) + 1 / (2 * p_res_y))).tolist()
    cmap = plt.cm.get_cmap("magma")
    y_to_color = (torch.tensor(list(map(cmap, y_steps)), device=device)[:, :3] * 255).to(torch.uint8)
    faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
    faces_t = torch.tensor(faces, device=device)

    print("Setup folders")
    os.makedirs(args.save_path, exist_ok=True)
    print('Loading checkpoint')
    cp = torch.load(args.model_path, map_location=device)

    net = KYN(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    _wrapper.load_state_dict(cp["model"], strict=False) if "model" in cp.keys() else _wrapper.load_state_dict(cp, strict=False)
    renderer.to(device)
    renderer.eval()

    ray_sampler = ImageRaySampler(config["model_conf"]["z_near"], config["model_conf"]["z_far"], *resolution, norm_dir=False)

    print("Load input image")
    assert os.path.exists(args.img)
    img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img = cv2.resize(img, (resolution[1], resolution[0]))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) * 2 - 1
    img_name = os.path.basename(args.img).split(".")[0]

    with torch.no_grad():

        poses = torch.eye(4).view(1, 1, 4, 4).to(device)
        projs = proj.view(1, 1, 4, 4).to(device)[:, :, :3, :3]

        net.encode(img, projs, poses, ids_encoder=[0], ids_render=[0])
        net.set_scale(0)

        q_pts = get_pts_vox(X_RANGE, Y_RANGE, Z_RANGE, p_res[1], p_res_y, p_res[0])
        q_pts = q_pts.to(device).reshape(1, -1, 3)
        _, invalid, sigmas = net.forward(q_pts)


        img_save = img[0, 0].permute(1, 2, 0).cpu() * .5 + .5
        save_plot(img_save.numpy(), os.path.join(args.save_path, f"{img_name}_in.png"), dry_run=False)

        # generate depth
        print(f"Generating depth of " + os.path.join(args.save_path, f"{img_name}"))
        _, depth = render_poses(renderer, ray_sampler, poses[:, :1], projs[:, :1])
        
        depth = ((1 / depth - 1 / config["model_conf"]["z_far"]) / (1 / config["model_conf"]["z_near"] - 1 / config["model_conf"]["z_far"])).clamp(0, 1)
        
        save_plot(color_tensor(depth, "magma", norm=True).numpy(), os.path.join(args.save_path, f"{img_name}_depth.png"), dry_run=False)

        # generate bev
        print(f"Generating BEV map of " + os.path.join(args.save_path, f"{img_name}"))
        render_range_dict["y_range"] = [0, 0.75]
        profile = render_profile(net, cam_incl_adjust, render_range_dict=render_range_dict)
        save_plot(color_tensor(profile.cpu(), "magma", norm=True).numpy(), os.path.join(args.save_path, f"{img_name}_bev.png"), dry_run=False)


        # generate voxel
        print(f"Generating voxel of " + os.path.join(args.save_path, f"{img_name}"))
        alphas = sigmas
        alphas = alphas.reshape(1, 1, p_res[1], p_res_y, p_res[0])              # (x, y, z)
        alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)            
        is_occupied = alphas_mean.squeeze() > .5
        verts, faces, colors = build_voxels(is_occupied.nonzero(), p_res[1], p_res_y, p_res[0], q_pts.squeeze(0).T, y_to_color, faces_t)

        verts_el, faces_el = parse_ply(verts, faces, colors)

        save_name = f"{img_name}_voxel.ply"
        PlyData([verts_el, faces_el]).write(os.path.join(args.save_path, f"{img_name}_voxel.ply"))




if __name__ == '__main__':
    main()
