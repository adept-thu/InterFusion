import torch
import torch.nn as nn


matrix = 

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        if 'pillar_features' in batch_dict:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1

            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_dict['spatial_features'] = batch_spatial_features
        else:
            lidar_pillar_features, lidar_coords = batch_dict['lidar_pillar_features'], batch_dict['lidar_voxel_coords']
            radar_pillar_features, radar_coords = batch_dict['radar_pillar_features'], batch_dict['radar_voxel_coords']
            lidar_batch_spatial_features = []
            radar_batch_spatial_features = []
            lidar_batch_size = lidar_coords[:, 0].max().int().item() + 1
            radar_batch_size = radar_coords[:, 0].max().int().item() + 1

            for lidar_batch_idx in range(lidar_batch_size):
                lidar_spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=lidar_pillar_features.dtype,
                    device=lidar_pillar_features.device)

                lidar_batch_mask = lidar_coords[:, 0] == lidar_batch_idx
                lidar_this_coords = lidar_coords[lidar_batch_mask, :]
                lidar_indices = lidar_this_coords[:, 1] + lidar_this_coords[:, 2] * self.nx + lidar_this_coords[:, 3]
                lidar_indices = lidar_indices.type(torch.long)
                lidar_pillars = lidar_pillar_features[lidar_batch_mask, :]
                lidar_pillars = lidar_pillars.t()
                lidar_spatial_feature[:, lidar_indices] = lidar_pillars
                lidar_batch_spatial_features.append(lidar_spatial_feature)
            for radar_batch_idx in range(radar_batch_size):
                radar_spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=radar_pillar_features.dtype,
                    device=radar_pillar_features.device)

                radar_batch_mask = radar_coords[:, 0] == radar_batch_idx
                radar_this_coords = radar_coords[radar_batch_mask, :]
                radar_indices = radar_this_coords[:, 1] + radar_this_coords[:, 2] * self.nx + radar_this_coords[:, 3]
                radar_indices = radar_indices.type(torch.long)
                radar_pillars = radar_pillar_features[radar_batch_mask, :]
                radar_pillars = radar_pillars.t()
                radar_spatial_feature[:, radar_indices] = radar_pillars
                radar_batch_spatial_features.append(radar_spatial_feature)

            lidar_batch_spatial_features = torch.stack(lidar_batch_spatial_features, 0)
            radar_batch_spatial_features = torch.stack(radar_batch_spatial_features, 0)
            lidar_batch_spatial_features = lidar_batch_spatial_features.view(lidar_batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            radar_batch_spatial_features = radar_batch_spatial_features.view(radar_batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_dict['lidar_spatial_features'] = lidar_batch_spatial_features
            batch_dict['radar_spatial_features'] = radar_batch_spatial_features

            batch_dict['spatial_features'] = torch.cat((lidar_batch_spatial_features, radar_batch_spatial_features), 1)
        return batch_dict
