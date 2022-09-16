import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        # 得到anchor在点云中的分布范围
        # Get the distribution range of anchor in the point cloud.
        self.anchor_range = anchor_range
        # 得到配置参数中所有尺度anchor的长宽高
        # Get the length, width and height of the anchor for all scales in the configuration parameters.
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        # 得到anchor的旋转角度，这里表示为角度，即0度和90度
        # Get the rotation angle of the anchor, including 0 degrees or 90 degrees.
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        # 得到每个anchor初始化在点云中z轴的位置，其中在kitti中点云的z轴范围是-3m到1m
        # Get the coordinate position of each anchor in the z-axis direction in the point cloud after initialization.
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        # 每个先验框产生的时候是否需要在每个格子的中间
        # 默认为False
        # Determines whether each generated precedence box needs to be in the center of each grid,
        # the default result is False.
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        # 初始化参数
        # initialize the parameters
        all_anchors = []
        num_anchors_per_location = []
        # 将三个类别的先验框逐类别生成
        # Generate a priori boxes for each of the three categories in sequence by category.
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            # 每个位置产生2个anchor，这里的2代表两个方向
            # Each position produces two anchors in different directions.
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            # 不需要对齐中心点来生成先验框
            # There is no need to align the center points to generate a priori boxes.
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                # 中心对齐，然后平移半个网格
                # Align the center of the a priori box with the center of the grid it is on in one direction,
                # and then translate half the grid.
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                # 计算那个每个网格在点云空间中的实际大小
                # 用于将每个anchor映射回实际点云中的大小
                # The calculation is performed to obtain the actual size of each mesh in the point cloud space,
                # and then the calculation results are used to map each anchor to the size in the actual point cloud.
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                # 由于没有进行中心对齐，所有每个点相对于左上角坐标的偏移量均为0
                # Since there is no center alignment operation,
                # the relative offset of each point with respect to the coordinates of the upper left vertex of the grid is 0.
                x_offset, y_offset = 0, 0

            # 生成单个维度x_shifts, y_shifts和z_shifts
            # 以x_stride为step，在self.anchor_range[0] + x_offset和self.anchor_range[3] + 1e-5，
            # 进而生成x坐标
            # Generate single dimensions x_shifts, y_shifts, z_shifts.
            # generate the x-value of the coordinates
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            # 生成y坐标
            # generate the y-value of the coordinates
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            # 生成z坐标
            # generate the z-value of the coordinates
            z_shifts = x_shifts.new_tensor(anchor_height)

            # num_anchor_size = 1
            # num_anchor_rotation = 2
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            # [0, 1.57] 弧度制
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)

            # 调用meshgrid来生成网格坐标，即在原来的维度上进行扩展
            # The meshgrid is called to generate the grid coordinates, i.e., to expand on the original dimensions.
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            # 将各类别的anchor的z轴方向从anchor的底部移动到该anchor的中心点位置
            # Moves the z-axis direction of each category of anchor from the bottom of the anchor to the center point of that anchor.
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
