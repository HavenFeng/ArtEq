import argparse
import os

import numpy as np
import torch
import tqdm
import trimesh
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean

from amass_ptc_loader import AMASSDataset
from backbones import get_part_seg_loss
from geometry import (
    aug_so3_ptc,
    batch_rodrigues,
    get_body_model,
    rotation_matrix_to_angle_axis,
)
from loss_func import NormalVectorLoss, marker_loss, norm_loss
from models_pointcloud import PointCloud_network_equiv

torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=12)


# fmt: off
markers_idx = torch.tensor([3470, 3171, 3327, 857, 1812, 628, 182, 3116, 3040, 239,
                            1666, 1725, 0, 2174, 1568, 1368, 3387, 2112, 1053, 1058,
                            3336, 3346, 1323, 2108, 3122, 3314, 1252, 1082, 1861, 1454,
                            850, 2224, 3233, 1769, 6728, 4343, 5273, 4116, 3694, 6399,
                            6540, 6488, 3749, 5135, 5194, 3512, 5635, 5210, 4360, 4841,
                            6786, 5573, 4538, 4544, 6736, 6747, 4804, 5568, 6544, 6682,
                            5322, 4927, 5686, 4598, 6633, 3506, 3508])
# fmt: on


def to_categorical(y, num_classes):
    """1-hot encodes a tensor."""
    new_y = torch.eye(num_classes, device=y.device)[y]
    return new_y


def local_to_global_bone_transformation(local_bone_transformation, parents):
    B, K = local_bone_transformation.shape[:2]
    local_bone_transformation = F.normalize(local_bone_transformation, dim=-2).double()

    Rs = [local_bone_transformation[:, 0]]
    for i in range(1, 22):
        Rs.append(torch.bmm(Rs[parents[i]], local_bone_transformation[:, i]))

    return torch.stack(Rs, dim=1).float()


def kinematic_layer_SO3_v2(global_bone_transformation, parents):
    """Input per-part global transformation Output local bone transformation based on SMPL kinematic tree, rotational
    information (SO3) as SMPL joint params translational info can be further aligned with SMPL meshes?"""

    B, K, D = global_bone_transformation.size()
    assert D != 6
    global_bone_transformation = global_bone_transformation.reshape(B, K, 3, 3)

    root_joint = global_bone_transformation[:, 0:1]
    joint_indices = torch.arange(1, 22)
    R_global = global_bone_transformation[:, joint_indices]
    R_parent = global_bone_transformation[:, parents[1:]]
    R_local = torch.bmm(
        torch.transpose(R_parent, 2, 3).reshape(B * (K - 1), 3, 3), R_global.reshape(B * (K - 1), 3, 3)
    ).reshape(B, K - 1, 3, 3)

    return torch.cat([root_joint, R_local], 1).reshape(B, K, 3, 3)


def get_nc_and_view_channel(args):
    if args.rep_type == "aa":
        nc = 3
    elif args.rep_type == "6d":
        nc = 6

    view_channel = 4

    return nc, view_channel


def trimesh_sampling(vertices, faces, count):
    body_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
    _, sample_face_idx = trimesh.sample.sample_surface_even(body_mesh, count)
    if sample_face_idx.shape[0] != count:
        print("add more face idx to match num_point")
        missing_num = count - sample_face_idx.shape[0]
        add_face_idx = np.random.choice(sample_face_idx, missing_num)
        sample_face_idx = np.hstack((sample_face_idx, add_face_idx))
    r = np.random.rand(count, 2)

    A = vertices[:, faces[sample_face_idx, 0], :]
    B = vertices[:, faces[sample_face_idx, 1], :]
    C = vertices[:, faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    lbs_w = gt_lbs.cpu().numpy()
    A_lbs = lbs_w[faces[sample_face_idx, 0], :]
    B_lbs = lbs_w[faces[sample_face_idx, 1], :]
    C_lbs = lbs_w[faces[sample_face_idx, 2], :]
    P_lbs = (
        (1 - np.sqrt(r[:, 0:1])) * A_lbs
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B_lbs
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C_lbs
    )

    return P, P_lbs


def sample_points(vertices, faces, count, fix_sample=False, sample_type="trimesh"):
    assert not fix_sample and sample_type == "trimesh"
    return trimesh_sampling(vertices, faces, count)


def get_pointcloud(vertices, n_points_surface, points_sigma):
    points_surface, points_surface_lbs = sample_points(
        vertices=vertices.cpu().numpy(), faces=body_model_faces, count=n_points_surface
    )

    points_surface = torch.from_numpy(points_surface).to(vertices.device)
    points_surface_lbs = torch.from_numpy(points_surface_lbs).to(vertices.device)

    labels = get_joint_label_merged(points_surface_lbs)

    points_surface += points_sigma * torch.randn(points_surface.shape[1], 3).to(vertices.device)
    points_label = labels[None, :].repeat(vertices.size(0), 1)

    return points_surface.float(), points_label, points_surface_lbs


def get_joint_label_merged(lbs_weights):
    gt_joint = torch.argmax(lbs_weights, dim=1)
    gt_joint1 = torch.where((gt_joint == 22), 20, gt_joint)
    gt_joint2 = torch.where((gt_joint1 == 23), 21, gt_joint1)
    gt_joint2 = torch.where((gt_joint2 == 10), 7, gt_joint2)
    gt_joint2 = torch.where((gt_joint2 == 11), 8, gt_joint2)

    return gt_joint2


def SMPLX_layer(body_model, betas, translation, motion_pose, rep="6d"):
    bz = body_model.batch_size

    if rep == "rotmat":
        motion_pose_aa = rotation_matrix_to_angle_axis(motion_pose.reshape(-1, 3, 3)).reshape(bz, -1)
    else:
        motion_pose = motion_pose.squeeze().reshape(bz, -1)
        motion_pose_aa = motion_pose

    zero_center = torch.zeros_like(translation.reshape(-1, 3).cuda())
    body_param_rec = {}
    body_param_rec["transl"] = zero_center
    body_param_rec["global_orient"] = motion_pose_aa[:, :3].cuda()
    body_param_rec["body_pose"] = torch.cat([motion_pose_aa[:, 3:66].cuda(), torch.zeros(bz, 6).cuda()], dim=1)
    body_param_rec["betas"] = betas.reshape(bz, -1)[:, :10].cuda()

    body_mesh = body_model(return_verts=True, **body_param_rec)
    mesh_rec = body_mesh.vertices
    mesh_j_pose = body_mesh.joints
    return mesh_j_pose, mesh_rec


def train(args, model, body_model, optimizer, train_loader):
    model.train()

    pbar = tqdm.tqdm(train_loader)
    for batch_data in pbar:
        motion_pose_aa = batch_data["rotations"].to(args.device)

        if args.aug_type == "so3":
            global_root = motion_pose_aa[:, 0]
            global_root_aug = aug_so3_ptc(global_root)
            motion_pose_aa[:, 0] = global_root_aug

        motion_trans = batch_data["translation"].to(args.device)
        B, _ = motion_trans.size()

        motion_pose_rotmat = batch_rodrigues(motion_pose_aa.reshape(-1, 3)).reshape(B, -1, 3, 3)
        motion_pose_rotmat_global = local_to_global_bone_transformation(motion_pose_rotmat, parents)

        betas = batch_data["body_shape"][:, None, :].to(args.device)

        gt_joints_pos, gt_vertices = SMPLX_layer(body_model, betas, motion_trans, motion_pose_rotmat, rep="rotmat")

        pcl_data, label_data, pcl_lbs = get_pointcloud(gt_vertices, args.num_point, points_sigma=0.001)
        optimal_trans = False

        losses = {}

        if epoch < 1:
            gt_part_seg = to_categorical(label_data, 22).cuda()
        else:
            gt_part_seg = None

        gt_bipart = None

        pred_joint, pred_pose, pred_shape, trans_feat = model(
            pcl_data, gt_part_seg, gt_bipart, pcl_lbs, is_optimal_trans=optimal_trans, parents=parents
        )

        gt_joints_set = [10, 11]

        pred_pose[:, gt_joints_set] = motion_pose_rotmat_global.reshape(pred_pose.shape[0], pred_pose.shape[1], -1)[
            :, gt_joints_set
        ]

        pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)

        pose_params_loss_global = norm_loss(pred_pose[0, :22], motion_pose_rotmat_global[0, :22], loss_type="l2")

        angle_loss = pose_params_loss_global

        pred_joints_pos, pred_vertices = SMPLX_layer(
            body_model, pred_shape, motion_trans, pred_joint_pose, rep="rotmat"
        )

        pred_pcl_part_mean = model.soft_aggr_norm(pcl_data.unsqueeze(3), pred_joint).squeeze()
        gt_pcl_part_mean = scatter_mean(pcl_data, label_data.cuda(), dim=1)

        losses["seg_loss"] = (
            point_cls_loss(
                pred_joint.contiguous().view(-1, pred_joint.shape[-1]),
                label_data.reshape(
                    -1,
                ),
                trans_feat,
            )
            * args.part_w
        )

        losses["angle_recon"] = angle_loss * args.angle_w

        losses["beta"] = F.mse_loss(pred_shape, betas.reshape(B, -1)[:, :10])

        losses["joints_pos"] = (
            F.mse_loss(gt_joints_pos.reshape(-1, 45, 3), pred_joints_pos.reshape(-1, 45, 3)) * args.jpos_w
        )

        losses["vertices"] = F.mse_loss(gt_vertices, pred_vertices) * args.vertex_w

        losses["normal"] = surface_normal_loss(pred_vertices, gt_vertices).mean() * args.normal_w

        losses["marker"] = (
            marker_loss(verts_pred=pred_vertices, verts_gt=gt_vertices, markers=markers_idx) * args.vertex_w * 2
        )

        losses["pcl_part_mean"] = F.mse_loss(pred_pcl_part_mean, gt_pcl_part_mean) * args.vertex_w

        all_loss = 0.0
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        if all_loss.isnan().any():
            print("Loss is NaN issue")
            all_loss = 0.0

        losses["all_loss"] = all_loss

        optimizer.zero_grad()

        all_loss.backward()
        for nm, param in model.named_parameters():
            if param.grad is None:
                pass
            elif param.grad.sum().isnan():
                param.grad = torch.zeros_like(param.grad)

        optimizer.step()

        pred_choice = pred_joint.clone().reshape(-1, part_num).data.max(1)[1]
        correct = (
            pred_choice.eq(
                label_data.reshape(
                    -1,
                ).data
            )
            .cpu()
            .sum()
        )
        batch_correct = correct.item() / (args.batch_size * args.num_point)

        pbar.set_description(f"Batch part acc: {batch_correct:.03f}")
    return all_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument(
        "--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument("--epochs", type=int, default=15, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="N", help="learning rate")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
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
    parser.add_argument("--part_w", type=float, default=5, help="")
    parser.add_argument("--angle_w", type=float, default=5, help="")
    parser.add_argument("--jpos_w", type=float, default=1e2, help="")
    parser.add_argument("--vertex_w", type=float, default=1e2, help="")
    parser.add_argument("--normal_w", type=float, default=1e0, help="")
    parser.add_argument("--i", type=int, default=None, help="")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda

    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.cuda else "cpu")

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

    part_num = args.part_num
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nc, _ = get_nc_and_view_channel(args)

    model = PointCloud_network_equiv(
        option=args,
        z_dim=args.latent_num,
        nc=nc,
        part_num=part_num,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = AMASSDataset()

    body_model = get_body_model(model_type="smpl", gender="male", batch_size=args.batch_size, device="cuda")

    body_model_faces = body_model.faces.astype(int)

    parents = body_model.parents[:22]

    gt_lbs = body_model.lbs_weights

    surface_normal_loss = NormalVectorLoss(face=body_model.faces.astype(int))
    point_cls_loss = get_part_seg_loss()

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    for epoch in range(args.epochs):
        average_all_loss = train(args, model, body_model, optimizer, train_loader)

        torch.save(
            model.state_dict(),
            os.path.join(output_folder, f"model_epochs_{epoch:08d}.pth"),
        )
