import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F

from EPN_options import get_default_cfg
from layers import BatchMLP
from pointnet2_utils import PointFeatPropagation
from so3conv import so3_mean


class PointCloud_network_equiv(nn.Module):
    """Single frame pointcloud encoding network, no temporal aggregration."""

    def __init__(
        self,
        option=None,
        z_dim=10,
        nc=3,
        part_num=24,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.part_num = part_num
        self.kinematic_cond = option.kinematic_cond

        """
        EPN: KPConv+Icosahedron
        """

        from backbones import ResnetPointnet
        from layers import StackedMHSA
        from so3net import build_model

        model_setting_file = "./experiments/EPN_model_setting.json"
        EPN_cfg = get_default_cfg()
        EPN_cfg.model.search_radius = option.EPN_input_radius

        mlp_layers = [[32, 32], [64, 64], [128, 128], [256, 256]]
        strides_layers = [2, 2, 2, 2]

        EPN_layer_n = option.EPN_layer_num
        EPN_feat_dim = mlp_layers[EPN_layer_n - 1][0]
        if self.kinematic_cond == "yes":
            EPN_feat_dim2 = EPN_feat_dim * 2
        else:
            EPN_feat_dim2 = EPN_feat_dim

        self.encoder = build_model(
            EPN_cfg, mlps=mlp_layers[:EPN_layer_n], strides=strides_layers[:EPN_layer_n], to_file=model_setting_file
        )
        self.shape_encoder = StackedMHSA(embedding_dim=EPN_feat_dim2, value_dim=6, num_heads=8, num_layers=2)
        self.pose_encoder = StackedMHSA(embedding_dim=EPN_feat_dim2, value_dim=128, num_heads=8, num_layers=2)
        self.part_encoder = ResnetPointnet(out_dim=self.part_num, hidden_dim=128, dim=EPN_feat_dim)

        self.shape_predictor = BatchMLP(in_features=22 * 6, out_features=10)
        self.pose_predictor = BatchMLP(in_features=128, out_features=128)
        self.so3_reg = nn.Conv1d(128, 1, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def encode(self, f):
        return self.encoder(f)

    def decode_shape(self, inv_feat):
        """
        input: B, part_num, 256
        output: B, 10
        """

        B, part_num, feat_dim = inv_feat.shape
        x = self.shape_encoder(inv_feat)
        body_x = x.reshape(B, -1)
        pred_shape = self.shape_predictor(body_x)
        return pred_shape

    def decode_pose(self, equiv_feat, anchors):
        """
        input: B, part_num, 60, 256
        output: B, part_num, 6
        """
        B, part_num, feat_dim, na = equiv_feat.shape

        x = self.pose_encoder(equiv_feat.permute(0, 1, 3, 2).reshape(-1, na, feat_dim).contiguous())
        x = self.pose_predictor(x)
        anc_w = self.so3_reg(x.permute(0, 2, 1))

        pred_pose = so3_mean(anchors[None, ...].repeat(B * part_num, 1, 1, 1), anc_w.squeeze())
        return pred_pose.reshape(B, part_num, 9)

    def soft_aggr_norm(self, feat, part_seg):
        """This is using softmax on the predicted weight to aggregate the feature."""

        B, N, part_num = part_seg.shape
        part_seg = part_seg.permute(0, 2, 1)
        part_feat_list = []
        for part_i in range(part_num):
            feat_weight = part_seg[:, part_i, :]
            feat_weight_normalized = F.normalize(feat_weight, p=1, dim=1)
            weighted_feat = feat_weight_normalized[..., None, None] * feat
            part_i_feat = weighted_feat.sum(1)
            part_feat_list.append(part_i_feat.unsqueeze(1))

        part_feat = torch.cat(part_feat_list, dim=1)

        return part_feat

    def kinematic_conditioning(self, point_equiv, part_equiv, parents):
        part_equiv_parent = part_equiv[:, parents[1:]]
        root_equiv = part_equiv[:, 0:1]
        cond_feat = torch.cat([root_equiv, part_equiv_parent], dim=1)

        part_equiv_cond = torch.cat([cond_feat, part_equiv], dim=2)

        return part_equiv_cond

    def forward(
        self, f_data, gt_part_seg=None, gt_bipart=None, pcl_lbs=None, debug=False, is_optimal_trans=False, parents=None
    ):
        B, N, C = f_data.size()

        r = self.encode(f_data)

        equiv_feat_xyz = r.xyz
        S = equiv_feat_xyz.shape[-1]
        equiv_feat = r.feats.permute(0, 1, 3, 2).reshape(B, -1, S)
        so3_anchors = r.anchors

        point_equiv_feat = PointFeatPropagation(
            xyz1=f_data.permute(0, 2, 1), xyz2=equiv_feat_xyz, points2=equiv_feat
        ).reshape(B, N, -1, 60)
        point_inv_feat = point_equiv_feat.mean(-1)

        part_seg = self.part_encoder(point_inv_feat)
        if gt_part_seg is not None:
            part_seg_info = gt_part_seg
        else:
            part_seg_info = part_seg

        part_equiv_feat = self.soft_aggr_norm(point_equiv_feat, part_seg_info)
        if self.kinematic_cond == "yes":
            part_equiv_feat = self.kinematic_conditioning(point_equiv_feat, part_equiv_feat, parents)

        part_inv_feat = part_equiv_feat.mean(-1)
        pred_shape = self.decode_shape(part_inv_feat)
        pred_pose = self.decode_pose(part_equiv_feat, so3_anchors)

        trans_feat = None

        return part_seg, pred_pose, pred_shape, trans_feat
