import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

# This section is similar to a simplified version of PointNet.
class interRAL(nn.Module):
    def __init__(self, channels):
        super(interRAL, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)      # ascend dimension with 64 output channels
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)   # 1D BN layer
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)     # ascend the demension
        torch.backends.cudnn.enabled = False
        # change dimensions, (pillars,num_points,channels) -> (pillars,channels,num_points)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # maxpool operation, and find the point in each pillar that best represents the pillar
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            # return the results obtained by processing the pillar from a simplified version of PointNet
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL(64)    # set the channel number of interRAL

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'],
                                                    batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) /
                        voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features, radar_f_cluster, radar_f_center]
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()

            # safusionlayer2
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features)
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict
