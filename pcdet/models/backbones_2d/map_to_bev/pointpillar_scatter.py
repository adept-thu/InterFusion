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
            # 拿到经过PointNet处理后的pillar数据和每个pillar艘在点云中的坐标位置
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

            # 将转换为伪图像的数据存到该列表中
            # Store the data converted to pseudo-images in this list.
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1

            # 对batch中的每个数据独立进行处理
            # Process each piece of data in the batch sequentially and independently.
            for batch_idx in range(batch_size):
                # 穿件一个空间坐标，用来接收pillar中的数据
                # Create a spatial coordinate to receive the data from the pillar.
                # self.num_bev_features为64
                # self.num_bev_features: the value is 64
                # self.nz * self.nx *self.ny是生成的空间坐标索引的乘积
                # self.nz * self.nx * self.ny: the product of the generated spatial coordinate indices
                # pillar feature:
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                # 从coords[:, 0]提取该batch_idx的数据mask
                # Extract the data mask of batch_idx from coords[:,0]
                batch_mask = coords[:, 0] == batch_idx
                # 提取mask提取坐标
                # Extracting coordinate information based on mask
                this_coords = coords[batch_mask, :]
                # 获取所有非空pillar在伪图像的对应索引位置
                # Get the position of all the indexes corresponding to the non-empty pillar on the pseudo-image.
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                # 转换数据类型
                # converting data types
                indices = indices.type(torch.long)
                # 根据mask提取pillar_features
                # Extract the pillar_feature based on mask.
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                # 在索引位置填充pillars
                # Fill in the index position with pillars.
                spatial_feature[:, indices] = pillars
                # 将空间特征加入list
                # Add spatial features to the list.
                batch_spatial_features.append(spatial_feature)

            # 在第0个维度将所有的数据堆叠在一起
            # Stack all data in dimension 0
            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            # reshape回原空间（伪图像）
            # reshape back to the original space, i.e. pseudo-image
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            # 将结果加入batch_dict
            # Add the results to batch_dict
            batch_dict['spatial_features'] = batch_spatial_features
        else:
            # 对于融合的算法，按照需要对不同模态的数据分别进行处理。
            # Process the information of different modalities in sequence and generate the results.
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
