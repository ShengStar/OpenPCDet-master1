from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock_9(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dilation=1):
        super(SparseBasicBlock_9, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=(1,9), stride=stride, padding=8, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=(9,1), stride=stride, padding=8, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None: # 不执行
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out



class SparseBasicBlock_dilation(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dilation=1):
        super(SparseBasicBlock_dilation, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key, dilation = dilation
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dilation=1):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

        # self.conv_dila = spconv.SubMConv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=1, bias=bias)
        # self.bn_dila = norm_fn(planes)
        # self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        # out = self.conv_dila(x)
        # out = replace_feature(out, self.bn_dila(out.features))
        # out_dila = replace_feature(out, self.relu(out.features))

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None: # 不执行
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xVoxelNeXt2D(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.relu = nn.ReLU()

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1', dilation=1),
            # SparseBasicBlock_9(32, 32, norm_fn=norm_fn, indice_key='res11', dilation=1),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1', dilation=1),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1', dilation=1),
        )
        self.conv1_3 = spconv.SparseSequential(SparseBasicBlock_dilation(32, 32, norm_fn=norm_fn, indice_key='res13', dilation=3))
        self.conv1_6 = spconv.SparseSequential(SparseBasicBlock_dilation(32, 32, norm_fn=norm_fn, indice_key='res13', dilation=6))
        self.conv1_9 = spconv.SparseSequential(SparseBasicBlock_dilation(32, 32, norm_fn=norm_fn, indice_key='res13', dilation=9))
        self.conv1_12 = spconv.SparseSequential(SparseBasicBlock_dilation(32, 32, norm_fn=norm_fn, indice_key='res13', dilation=12))


        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2', dilation=1),
            # SparseBasicBlock_9(64, 64, norm_fn=norm_fn, indice_key='res21', dilation=1),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2', dilation=1),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2', dilation=1),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2', dilation=1),
        )

        self.conv2_3 = spconv.SparseSequential(SparseBasicBlock_dilation(64, 64, norm_fn=norm_fn, indice_key='res23', dilation=3))
        self.conv2_6 = spconv.SparseSequential(SparseBasicBlock_dilation(64, 64, norm_fn=norm_fn, indice_key='res23', dilation=6))
        self.conv2_9 = spconv.SparseSequential(SparseBasicBlock_dilation(64, 64, norm_fn=norm_fn, indice_key='res23', dilation=9))
        self.conv2_12 = spconv.SparseSequential(SparseBasicBlock_dilation(64, 64, norm_fn=norm_fn, indice_key='res23', dilation=12))

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3', dilation=1),
        )

        self.conv3_3 = spconv.SparseSequential(SparseBasicBlock_dilation(128, 128, norm_fn=norm_fn, indice_key='res33', dilation=3))
        self.conv3_6 = spconv.SparseSequential(SparseBasicBlock_dilation(128, 128, norm_fn=norm_fn, indice_key='res33', dilation=6))
        self.conv3_9 = spconv.SparseSequential(SparseBasicBlock_dilation(128, 128, norm_fn=norm_fn, indice_key='res33', dilation=9))
        self.conv3_12 = spconv.SparseSequential(SparseBasicBlock_dilation(128, 128, norm_fn=norm_fn, indice_key='res33', dilation=12))

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4', dilation=1),
            # SparseBasicBlock_9(256, 256, norm_fn=norm_fn, indice_key='res41', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4', dilation=1),
        )

        self.conv4_3 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res43', dilation=3))
        self.conv4_6 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res43', dilation=6))
        self.conv4_9 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res43', dilation=9))
        self.conv4_12 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res43', dilation=12))

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5', dilation=1),
            # SparseBasicBlock_9(256, 256, norm_fn=norm_fn, indice_key='res51', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5', dilation=1),
        )

        self.conv5_3 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res53', dilation=3))
        self.conv5_6 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res53', dilation=6))
        # self.conv5_12 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res53', dilation=12))
        # self.conv5_18 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res53', dilation=18))


        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6', dilation=1),
            # SparseBasicBlock_9(256, 256, norm_fn=norm_fn, indice_key='res61', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6', dilation=1),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6', dilation=1),
        )

        self.conv6_3 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res63', dilation=3))
        # self.conv6_6 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res63', dilation=6))
        # self.conv6_12 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res63', dilation=12))
        # self.conv6_18 = spconv.SparseSequential(SparseBasicBlock_dilation(256, 256, norm_fn=norm_fn, indice_key='res63', dilation=18))

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv.spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        print(input_sp_tensor.features.sparse_shape)

        x_conv1 = self.conv1(input_sp_tensor)

        conv1_3 = self.conv1_3(x_conv1)
        conv1_6 = self.conv1_6(x_conv1)
        conv1_9 = self.conv1_9(x_conv1)
        conv1_12 = self.conv1_12(x_conv1)
        x_conv1 = replace_feature(x_conv1, x_conv1.features + conv1_3.features + conv1_6.features + conv1_9.features + conv1_12.features)
        x_conv1 = replace_feature(x_conv1, self.relu(x_conv1.features))

        x_conv2 = self.conv2(x_conv1)

        conv2_3 = self.conv2_3(x_conv2)
        conv2_6 = self.conv2_6(x_conv2)
        conv2_9 = self.conv2_9(x_conv2)
        conv2_12 = self.conv2_12(x_conv2)
        x_conv2 = replace_feature(x_conv2, x_conv2.features + conv2_3.features + conv2_6.features + conv2_9.features + conv2_12.features)
        x_conv2 = replace_feature(x_conv2, self.relu(x_conv2.features))

        x_conv3 = self.conv3(x_conv2)

        conv3_3 = self.conv3_3(x_conv3)
        conv3_6 = self.conv3_6(x_conv3)
        conv3_9 = self.conv3_9(x_conv3)
        conv3_12 = self.conv3_12(x_conv3)
        x_conv3 = replace_feature(x_conv3, x_conv3.features + conv3_3.features + conv3_6.features + conv3_9.features + conv3_12.features)
        x_conv3 = replace_feature(x_conv3, self.relu(x_conv3.features))

        x_conv4 = self.conv4(x_conv3)

        conv4_3 = self.conv4_3(x_conv4)
        conv4_6 = self.conv4_6(x_conv4)
        conv4_9 = self.conv4_9(x_conv4)
        conv4_12 = self.conv4_12(x_conv4)
        x_conv4 = replace_feature(x_conv4, x_conv4.features + conv4_3.features + conv4_6.features + conv4_9.features + conv4_12.features)
        x_conv4 = replace_feature(x_conv4, self.relu(x_conv4.features))

        x_conv5 = self.conv5(x_conv4)

        conv5_3 = self.conv5_3(x_conv5)
        conv5_6 = self.conv5_6(x_conv5)
        # conv5_12 = self.conv5_12(x_conv5)
        # conv5_18 = self.conv5_18(x_conv5)
        # x_conv5 = replace_feature(x_conv5, x_conv5.features + conv5_3.features + conv5_6.features + conv5_12.features + conv5_18.features)
        x_conv5 = replace_feature(x_conv5, x_conv5.features + conv5_3.features + conv5_6.features)

        x_conv5 = replace_feature(x_conv5, self.relu(x_conv5.features))

        x_conv6 = self.conv6(x_conv5)

        conv6_3 = self.conv6_3(x_conv6)
        # conv6_6 = self.conv6_6(x_conv6)
        # conv6_12 = self.conv6_12(x_conv6)
        # conv6_18 = self.conv6_18(x_conv6)
        # x_conv6 = replace_feature(x_conv6, x_conv6.features + conv6_3.features + conv6_6.features + conv6_12.features + conv6_18.features)
        x_conv6 = replace_feature(x_conv6, x_conv6.features + conv6_3.features)

        x_conv6 = replace_feature(x_conv6, self.relu(x_conv6.features))

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        
        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict
