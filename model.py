"""Network architectures: RRWNet (Task 1) and CMRRWNet (Task 2).

PROVENANCE
----------
This file is DERIVED from the official GAVE2 baseline
(https://github.com/Peng2004/CMRRWNet), which itself builds on RRWNet
(Morano et al., 2024, https://github.com/j-morano/rrwnet). Third-party code is
kept close to the original so that the diff against upstream stays readable;
comments originally written in Chinese by the upstream authors have been
translated to English, without any change in behaviour.

OUR MODIFICATIONS
-----------------
One functional change, applied in two places (`RRWNet.forward` and
`CMRRWNet.forward`): the **topological-anchor fix**.

RRWNet refines the artery/vein maps iteratively while holding the vessel
(binary vasculature) channel fixed as an "anchor". Upstream RRWNet orders its
channels [artery, vein, vessel], so anchoring index 2 is correct there. GAVE2
re-ordered the convention to [artery, vessel, vein] for the losses and the
ground truth, but did NOT update the anchor index or the concatenation order.
The published baseline therefore anchors the **vein** channel instead of the
**vessel** channel, and iteratively refines artery-vs-vessel instead of
artery-vs-vein.

We reorder explicitly around the refinement sub-network (`_to_av_vessel` /
`_to_a_vessel_v`) so the anchor is the vessel channel, consistent with the loss
channel order (0=artery, 1=vessel, 2=vein).

Every `MODIFICATION` comment in this file marks a line we changed. A
line-by-line diff against upstream 

KNOWN LIMITATIONS
-----------------
* Only `RRWNet` and `CMRRWNet` are used by the final pipeline. The other classes
  (`UNet`, `WNet`, `RRUNet`, `RRWNetAll`, the SE/attention blocks) are retained
  from upstream because `CMRRWNet` composes some of them, and removing dead
  branches would make the diff against upstream harder to audit.
* `NewUNetModule` (the dual-encoder fusion trunk used by CMRRWNet) carries
  variable names inherited from an RGB/thermal fusion codebase upstream
  ("visible"/"secondary"); they refer here to CFP and FFA respectively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv_block = nn.Sequential(
            self.conv1,
            activf(inplace=True),
            self.conv2,
            activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2* base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x


class UNet(UNetModule):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__(input_ch, output_ch, base_ch)

    def forward(self, x):
        return [super().forward(x)]


class WNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, output_ch, base_ch)

    def forward(self, x):
        first_x = self.first_u(x)
        first_x_sig = torch.sigmoid(first_x)
        second_x = self.second_u(first_x_sig)
        return [first_x, second_x]


class RRUNet(nn.Module):
    """Mosinska et al. approach (without topology loss)"""

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.unet_module = UNetModule(input_ch+output_ch, output_ch, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []
        x_size = x.size()
        zero_maps_size = (
            x_size[0],
            3,  # only refine AV3
            x_size[2],
            x_size[3],
        )
        zero_maps = torch.zeros(zero_maps_size).to(x.device)
        x_maps = torch.cat((x, zero_maps), dim=1)

        pred = self.unet_module(x_maps)
        predictions.append(pred)

        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)
            x_maps = torch.cat((x, pred[:, :3, :, :]), dim=1)
            pred = self.unet_module(x_maps)
            predictions.append(pred[:, :3, :, :])

        return predictions


class RRWNetAll(nn.Module):
    """Network with all channels refined using a second recurrent
    UNet.
    """
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(3, 3, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        pred_1 = torch.sigmoid(pred_1)

        pred_2 = self.second_u(pred_1[:, :3, :, :])
        predictions.append(pred_2)

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = self.second_u(pred_2)
            predictions.append(pred_2)

        return predictions


class RRWNet(RRWNetAll):
    """RRWNetAll but refining only the A/V maps (the variant proposed in the paper).

    MODIFICATION (topological-anchor fix): `forward` below anchors the vessel
    channel (index 1 under the GAVE2 convention) instead of the vein channel
    (index 2, which is what the published baseline anchors). See the module
    docstring for the full rationale.
    """

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    @staticmethod
    def _to_av_vessel(p3):
        """Reorder [artery, vessel, vein] -> [artery, vein, vessel].

        Part of the topological-anchor fix (see MODIFICATION note in the class
        docstring): the refinement sub-network expects the anchor channel last,
        but GAVE2 stores vessel at index 1, not index 2.
        """
        return torch.cat([p3[:, 0:1], p3[:, 2:3], p3[:, 1:2]], dim=1)

    @staticmethod
    def _to_a_vessel_v(av2, vessel1):
        """Reassemble [artery, vein] + vessel anchor -> [artery, vessel, vein].

        Inverse of `_to_av_vessel`, restoring the GAVE2 channel order expected
        by the loss functions.
        """
        return torch.cat([av2[:, 0:1], vessel1, av2[:, 1:2]], dim=1)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        vessel_logits = pred_1[:, 1:2, :, :]
        pred_1_sig = torch.sigmoid(pred_1)
        vessel_prob = pred_1_sig[:, 1:2, :, :]

        refine_in = self._to_av_vessel(pred_1_sig)
        pred_2 = self.second_u(refine_in)
        predictions.append(self._to_a_vessel_v(pred_2, vessel_logits))

        for _ in range(self.num_iterations):
            av_prob = torch.sigmoid(pred_2)
            refine_in = torch.cat([av_prob, vessel_prob], dim=1)
            pred_2 = self.second_u(refine_in)
            predictions.append(self._to_a_vessel_v(pred_2, vessel_logits))

        return predictions

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, vis_fea):
        b, c, _, _ = vis_fea.size()
        y = self.avg_pool(vis_fea).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor
    
class ChannelAttentionConv(nn.Module):
    """Channel attention, 1D-convolution variant (ECA-style)."""

    def __init__(self, in_channel, gamma = 2, b = 1):
        """Initialise.

        Args:
            in_channel: number of channels of the input feature map.
            gamma: coefficient of the adaptive kernel-size formula.
            b: coefficient of the adaptive kernel-size formula.
        """
        super(ChannelAttentionConv, self).__init__()
        # Adapt the kernel size to the number of input channels.
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        # Force the kernel size to be odd.
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # Pooling.
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # 1D convolution.
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """Forward pass."""
        # Global pooling: [b,c,h,w] -> [b,c,1,1]
        avg_x = self.avg_pooling(X)
        max_x = self.max_pooling(X)
        # [b,c,1,1] -> [b,1,c] --1D conv--> [b,1,c] -> [b,c,1,1]
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Normalise the weights.
        v = self.sigmoid(avg_out + max_out)
        # Multiply the input feature map by the channel weights: [b,c,h,w]
        return  X * v
    
class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        # output_tensor = self.cSE(input_tensor) +  self.sSE(input_tensor)
        return output_tensor

    
class NewUNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

        # Encoder
        # RGB encoder (3 input channels: colour fundus photograph)
        self.conv1_rgb = ConvBlock(3, base_ch)
        self.conv2_rgb = ConvBlock(base_ch, 2 * base_ch)
        self.conv3_rgb = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4_rgb = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5_rgb = ConvBlock(8 * base_ch, 16 * base_ch)

        # A encoder (1 input channel: early-phase FFA)
        self.conv1_a = ConvBlock(1, base_ch)
        self.conv2_a = ConvBlock(base_ch, 2 * base_ch)
        self.conv3_a = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4_a = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5_a = ConvBlock(8 * base_ch, 16 * base_ch)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.attention1 = SELayer(base_ch)
        self.attention2 = SELayer(2 * base_ch)
        self.attention3 = SELayer(4 * base_ch)
        self.attention4 = SELayer(8 * base_ch)
        self.attention5 = SELayer(16 * base_ch)

        self.ca1 = ChannelAttentionConv(base_ch)
        self.ca2 = ChannelAttentionConv(2 * base_ch)
        # self.ca1 = ChannelSpatialSELayer(base_ch)
        # self.ca2 = ChannelSpatialSELayer(2 * base_ch)

        # self.ca3 = ChannelAttentionConv(8 * base_ch)
        # self.ca4 = ChannelAttentionConv(16 * base_ch)
        self.ca3 = ChannelSpatialSELayer(8 * base_ch)
        self.ca4 = ChannelSpatialSELayer(16 * base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)
    def Asy_fusion1(self, fea_rgb, fea_a, fea_av):  # smaller map first, larger map second
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv2_rgb(fea_rgb)
        add_fea = fea_a + fea_av + fea_rgb
        res = self.ca1(add_fea)
        return res
    def Asy_fusion2(self, fea_rgb, fea_a, fea_av):  # smaller map first, larger map second
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv3_rgb(fea_rgb)
        add_fea = fea_a + fea_av + fea_rgb
        res = self.ca2(add_fea)
        return res
    def Asy_fusion3(self, fea_rgb, fea_a, fea_av):  # smaller map first, larger map second
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv4_rgb(fea_rgb)
        add_fea = fea_a + fea_av + fea_rgb
        # add_fea = torch.cat([fea_a,fea_rgb], dim=1)
        res = self.ca3(add_fea)
        return res
    def Asy_fusion4(self, fea_rgb, fea_a, fea_av):  # smaller map first, larger map second
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv5_rgb(fea_rgb)
        add_fea = fea_a + fea_av + fea_rgb
        res = self.ca4(add_fea)
        return res
    
    def forward_features(self, feats_rgb, feats_a, feats_av):
        """
        x_rgb: B x C x H x W  #ct
        x_e  #pet
        """
        f0_0 = self.Asy_fusion1(feats_rgb[0], feats_a[1], feats_av[1])  # visible 0 (64) - secondary 1 (128)
        f1_0 = self.Asy_fusion2(feats_rgb[1], feats_a[2], feats_av[2])  # visible 1 (128) - secondary 2 (256)
        f2_0 = self.Asy_fusion3(feats_rgb[2], feats_a[3], feats_av[3])  # visible 2 (256) - secondary 3 (512)
        f3_0 = self.Asy_fusion4(feats_rgb[3], feats_a[4], feats_av[4])  # visible 3 (512) - secondary 4 (1024)

        return [f0_0, f1_0, f2_0, f3_0]
        return outs_fused
    
    def forward_encoder_rgb(self, rgb):
        x1 = self.conv1_rgb(rgb)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2_rgb(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3_rgb(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4_rgb(x)
        x = F.max_pool2d(x4, 2, 2)

        x5 = self.conv5_rgb(x)
        return [x1, x2, x3, x4, x5]

    # ======================
    # Encoder: A-modality forward pass
    # ======================
    def forward_encoder_a(self, a, feats_rgb):
        rgb_fea_x1, rgb_fea_x2, rgb_fea_x3, rgb_fea_x4, rgb_fea_x5 = feats_rgb
        x1 = self.conv1_a(a)
        x1 = self.attention1(x1, rgb_fea_x1)

        x2 = F.max_pool2d(x1, 2, 2)
        x2 = self.conv2_a(x2)
        x2 = self.attention2(x2, rgb_fea_x2)

        x3 = F.max_pool2d(x2, 2, 2)
        x3 = self.conv3_a(x3)
        x3 = self.attention3(x3, rgb_fea_x3)

        x4 = F.max_pool2d(x3, 2, 2)
        x4 = self.conv4_a(x4)
        x4 = self.attention4(x4, rgb_fea_x4)

        x5 = F.max_pool2d(x4, 2, 2)
        x5 = self.conv5_a(x5)
        x5 = self.attention5(x5, rgb_fea_x5)

        return [x1, x2, x3, x4, x5]

    
    def forward(self, x):
        rgb = x[:, 0:3, :, :]    # B x 3 x H x W
        a = x[:, 3:4, :, :]      # B x 1 x H x W  (keep the channel dim; do not squeeze to 3D)
        av = x[:, 4:5, :, :]     # B x 1 x H x W

        feats_rgb = self.forward_encoder_rgb(rgb)
        
        # 3. A Encoder
        feats_a = self.forward_encoder_a(a, feats_rgb)
        feats_av = self.forward_encoder_a(av, feats_rgb)
        
        # 4. Dual-modality feature fusion
        x1, x2, x3, x4= self.forward_features(feats_rgb, feats_a, feats_av)
        

        x = self.upconv1(x4)
        x = torch.cat((x3, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        # x = torch.cat((x1, x), dim=1)

        #x = self.conv9(x)
        x = self.outconv(x)

        return x


class CMRRWNet(RRWNet):
    """RRWNet with a cross-modal attention module in the first UNet.
    Proposed in the paper.
    """

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        self.first_u = NewUNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        # MODIFICATION (topological-anchor fix, same defect as in RRWNet above).
        # CMRRWNet reimplements forward()/refine() without going through
        # _to_av_vessel / _to_a_vessel_v, and therefore anchored index 2 (vein)
        # instead of index 1 (vessel). Fixed here to anchor the vessel channel,
        # consistent with the loss channel order (0=artery, 1=vessel, 2=vein).
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        vessel_logits = pred_1[:, 1:2, :, :]
        pred_1_sig = torch.sigmoid(pred_1)
        vessel_prob = pred_1_sig[:, 1:2, :, :]

        refine_in = self._to_av_vessel(pred_1_sig)
        pred_2 = self.second_u(refine_in)
        predictions.append(self._to_a_vessel_v(pred_2, vessel_logits))

        for _ in range(self.num_iterations):
            av_prob = torch.sigmoid(pred_2)
            refine_in = torch.cat([av_prob, vessel_prob], dim=1)
            pred_2 = self.second_u(refine_in)
            predictions.append(self._to_a_vessel_v(pred_2, vessel_logits))

        return predictions

    def refine(self, x):
        predictions = []
        vessel_prob = x[:, 1:2, :, :]
        vessel_logits = vessel_prob

        refine_in = self._to_av_vessel(x)
        pred_2 = self.second_u(refine_in)
        predictions.append(self._to_a_vessel_v(torch.sigmoid(pred_2), vessel_logits))

        for _ in range(self.num_iterations):
            av_prob = torch.sigmoid(pred_2)
            refine_in = torch.cat([av_prob, vessel_prob], dim=1)
            pred_2 = self.second_u(refine_in)
            predictions.append(self._to_a_vessel_v(torch.sigmoid(pred_2), vessel_logits))

        return predictions

def build_model(task: str, base_ch: int = 64, num_iterations: int = 5):
    """Instantiate the network for a given task with the shipped configuration.

    Parameters
    ----------
    task
        "task1" (CFP only, RRWNet, 3 input channels) or
        "task2" (CFP+FFA, CMRRWNet, 5 input channels).
    base_ch
        Width of the first convolution block (64 in all our runs).
    num_iterations
        Number of recursive refinement iterations (5 in all our runs).
    """
    if task == "task1":
        return RRWNet(input_ch=3, output_ch=3, base_ch=base_ch,
                      num_iterations=num_iterations)
    if task == "task2":
        return CMRRWNet(input_ch=5, output_ch=3, base_ch=base_ch,
                        num_iterations=num_iterations)
    raise ValueError(f"unknown task {task!r}, expected 'task1' or 'task2'")
