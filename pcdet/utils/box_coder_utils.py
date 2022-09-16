import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        # 截断anchor的[dx,dy,dz]，每个anchor的l,w,h数值如果小于1e-5，则应为1e-5
        # Truncate the [dx,dy,dz] of the anchors,
        # and if the value of each anchor_box parameter l, w, h is less than 1e-5,
        # then set these parameters to 1e-5.
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        # 截断boxes的[dx,dy,dz]，每个gt_boxes的l,w,h数值如果小于1e-5,则应为1e-5
        # Truncate the [dx,dy,dz] of the boxes,
        # and if the values of the parameters l, w, h of each GT_boxes are less than 1e-5,
        # set these parameters to 1e-5.
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        # 这里表明为torch.split的第二个参数。
        # split_size应为切分后每一块的大小，而不是切分为多少块
        # This is indicated as the second argument to torch.split.
        # torch.split(tensor, split_size, dim=), where split_size is the size of each piece after the cut,
        # and the extra parameter needs to be received using *cag.
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        # 计算anchor对角线的长度
        # calculate the diagonal length of the anchor
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        # 计算loss的公式
        # calculate the parameters △x,△y,△z,△w,△l,△h,△θ of loss by mathematical formula
        # In the calculation formula, g denotes GT and a denotes anchor.
        # △x = x ^ gt - xa ^ da
        xt = (xg - xa) / diagonal
        # △y = (y ^ gt - ya ^ da) / d ^ a
        yt = (yg - ya) / diagonal
        # △z = (z ^ gt - za ^ da) / h ^ a
        zt = (zg - za) / dza
        # △l = log(l ^ gt / l ^ a)
        dxt = torch.log(dxg / dxa)
        # △w = log(w ^ gt / w ^ a)
        dyt = torch.log(dyg / dya)
        # △h = log(h ^ gt / h ^ a)
        dzt = torch.log(dzg / dza)
        # False
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]     # △θ

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        # 这里指torch.split的第二个参数
        # This refers to the second argument of torch.split.
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        # 分割编码后的box PointPillars为False
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        # 计算anchor的对角线长度
        # calculate the diagonal length of the anchor
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        # loss计算中anchor与GT编码运算：g表示gt，a表示anchor
        # △x = (x^gt - xa^da)/diagonal --> x^gt = △x * diagonal + x^da
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        # △l = log(l^gt / l^a)的逆运算 --> l^gt = exp(△l) * l^a
        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        # 如果角度是cos和sin编码，采用新的解码方式 PointPillars为False
        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            # rts = [rg - ra] 角度的逆运算
            rg = rt + ra
        # PointPillars没有该项参数
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        # 这里表明为torch.split的第二个参数。
        # This is indicated as the second argument to torch.split.
        # torch.split(tensor, split_size, dim=)：这里的split_size是切分后每块的大小，对于多余的参数将使用*cags接收。
        # Here split_size is the size of each block after slicing,
        # for the extra parameters will be received using *cags.
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        # 计算anchor的对角线长度
        # calculate the diagonal length of the anchor
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        # 在计算公式中，g表示GT，a表示anchor。
        # In the calculation formula, g denotes GT and a denotes anchor.
        # △x = (x ^ gt = xa ^ da) / diagonal --> x ^ gt = △x * diagonal + x ^ da
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za
        # △l = log(l ^ gt / l ^ a) --> l ^ gt = exp(△l) * l ^ a
        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        # 如果角度的编码方式为sin和cos形式，应当选择使用新的解码形式。对于PointPillars，默认该形式为False。
        # If the angle is encoded in sin and cos form,
        # the new decoding form should be chosen to be used.
        # For PointPillars, the default form is False.
        rg = ra - rt
        # PointPillars默认为False。     # PointPillars defaults to False.
        cgs = [t + a for t, a in zip(cts, cas)]
        
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
