import argparse
import os
from pathlib import Path

import numpy as np
import smplx
import torch
import tqdm.auto as tqdm
import webdataset as wds

from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument(
        "--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument("--epoch", type=int, default=15, metavar="N", help="which model epoch to use (default: 15)")
    parser.add_argument("--rep_type", type=str, default="6d", metavar="N", help="aa, 6d")
    parser.add_argument("--part_num", type=int, default=22, metavar="N", help="part num of the SMPL body")
    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--aug_type", type=str, default="so3", metavar="N", help="so3, zrot, no")
    parser.add_argument("--gt_part_seg", type=str, default="auto", metavar="N", help="")
    parser.add_argument("--EPN_input_radius", type=float, default=0.4, help="train from pretrained model")
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument(
        "--kinematic_cond", type=str, default="yes", metavar="N", help="point num sampled from mesh surface"
    )
    parser.add_argument("--i", type=int, default=None, help="")
    parser.add_argument("--paper_model", action="store_true")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda

    args.device = torch.device("cuda")

    exps_folder = "gt_part_seg_{}_EPN_layer_{}_radius_{}_aug_{}_kc_{}".format(
        args.gt_part_seg,
        args.EPN_layer_num,
        args.EPN_input_radius,
        args.aug_type,
        args.kinematic_cond,
    )
    if args.num_point != 5000:
        exps_folder = exps_folder + f"_num_point_{args.num_point}"
    if args.i is not None:
        exps_folder = exps_folder + f"_{args.i}"

    output_folder = os.path.sep.join(["./experiments", exps_folder])

    if args.paper_model:
        output_folder = "./data/papermodel/"
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"

    nc, _ = get_nc_and_view_channel(args)

    torch.backends.cudnn.benchmark = True

    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device="cuda")
    parents = body_model.parents[:22]
    body_model_gt = smplx.SMPL("./data/smpl_model/SMPL_MALE.pkl", batch_size=1, num_betas=10)

    base_path = Path(output_folder)
    print(base_path)
    torch.manual_seed(1)

    model_path = base_path / f"model_epochs_{args.epoch-1:08d}.pth"

    model = PointCloud_network_equiv(option=args, z_dim=args.latent_num, nc=nc, part_num=args.part_num).to(args.device)

    model.load_state_dict(torch.load(model_path))

    for tar_folder in [Path("./data/DFaust_67_val/"), Path("./data/MPI_Limits/")]:
        print(
            f"{args.epoch=}, {args.aug_type=}, {args.kinematic_cond=} {tar_folder.name=} {args.EPN_layer_num=} {args.EPN_input_radius=}"
        )
        v2v = {}
        joint_err = {}
        acc = {}

        with torch.inference_mode():
            for tar_path in tqdm.tqdm(sorted(tar_folder.glob("*.tar"), key=lambda x: x.stem)):
                dataset = wds.WebDataset(str(tar_path)).decode().to_tuple("input.pth")
                for i, (batch,) in enumerate(dataset):
                    pcl_data = batch["pcl_data"][: args.num_point][None].cuda()
                    pred_joint, pred_pose, pred_shape, trans_feat = model(
                        pcl_data, None, None, None, is_optimal_trans=False, parents=parents
                    )

                    pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)

                    trans_feat = torch.zeros((1, 3)).cuda()

                    pred_joints_pos, pred_vertices = SMPLX_layer(
                        body_model,
                        pred_shape,
                        trans_feat,
                        pred_joint_pose,
                        rep="rotmat",
                    )
                    pred_joints_pos = pred_joints_pos[0].cpu()
                    pred_vertices = pred_vertices[0].cpu()

                    betas = batch["betas"][:10][None]
                    transl = torch.zeros_like(batch["trans"][None])
                    body_pose = batch["poses"][1:24].flatten()[None]
                    body_pose[:, -6:] = 0
                    global_orient = batch["poses"][0][None]

                    smpl_output = body_model_gt(
                        betas=betas, transl=transl, body_pose=body_pose, global_orient=global_orient
                    )
                    vertices_gt = smpl_output.vertices[0]
                    joints_gt = smpl_output.joints[0]

                    v2v[f"{tar_path.stem}_{i}"] = (
                        100 * (vertices_gt - pred_vertices).square().sum(dim=-1).sqrt().mean().item()
                    )
                    joint_err[f"{tar_path.stem}_{i}"] = (
                        100 * (joints_gt - pred_joints_pos).square().sum(dim=-1).sqrt().mean().item()
                    )
                    acc[f"{tar_path.stem}_{i}"] = (
                        (pred_joint[0].argmax(dim=1).cpu() == batch["label_data"][: args.num_point])
                        .mean(dtype=float)
                        .item()
                    )

        print(
            f"v2v={np.mean(list(v2v.values())):.3f} joint_err={np.mean(list(joint_err.values())):.3f} part_acc={100*np.mean(list(acc.values())):.3f}"
        )
