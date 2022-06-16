import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.cbam import CBAM

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class UNetConv(nn.Module):
    def __init__(self, in_planes, out_planes, att=True):
        super(UNetConv, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, 2, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

        if att:
            self.cbam = CBAM(out_planes, 16) # 这一步导致了通道数最低为128
        else:
            self.cbam = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_planes, out_planes, att=True):
        super(UpConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, in_planes//2, 4, 2, 1),
            nn.PReLU(in_planes//2),
        )
        
        # 也许不需要这么卷积，我不确定
        self.conv1 = conv(in_planes, in_planes//2, 3, 1, 1) 
        self.conv2 = conv(in_planes//2, out_planes, 3, 1, 1)

        if att:
            self.cbam = CBAM(out_planes, 16)
        else:
            self.cbam = None

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        y = self.conv1(torch.cat((x1, x2), 1))
        y = self.conv2(y)
        if self.cbam is not None:
            y = self.cbam(y)
        return y

class FeatureNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FeatureNet, self).__init__()
        # 处理IFBlock0时通道数问题
        self.conv0 = conv(7, in_planes, 1, 1, 0)
        
        self.conv1 = UNetConv(in_planes, out_planes//8, att=False)
        self.conv2 = UNetConv(out_planes//8, out_planes//4, att=True)
        self.conv3 = UNetConv(out_planes//4, out_planes//2, att=True)
        self.conv4 = UNetConv(out_planes//2, out_planes, att=True)
        self.conv5 = UNetConv(out_planes, 2*out_planes, att=True)

        self.deconv5 = UpConv(2*out_planes, out_planes, att=True)
        self.deconv4 = UpConv(out_planes, out_planes//2, att=False)
        self.deconv3 = UpConv(out_planes//2, out_planes//4, att=False)

    def forward(self, x, level=0):
        if x.shape[1] != 17:
            x = self.conv0(x)
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)
        x32 = self.conv5(x16)
        y = self.deconv5(x32, x16) # 匹配IFBlock0通道和尺寸

        # “早退机制”以期待用同一个UNet提取特征，不确定是否对训练产生影响
        if level != 0:
            y = self.deconv4(y, x8) # 匹配IFBlock1通道和尺寸
            if level == 2:
                y = self.deconv3(y, x4) # 匹配IFBlock2通道和尺寸
        return y

class IFBlock(nn.Module):
    def __init__(self, c=64, level=0):
        super(IFBlock, self).__init__()
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )       
        self.flowconv = nn.Conv2d(c, 4, 3, 1, 1)
        self.maskconvx16 = nn.Conv2d(c, 16*16*9, 1, 1, 0)
        self.maskconvx8 = nn.Conv2d(c, 8*8*9, 1, 1, 0)
        self.maskconvx4 = nn.Conv2d(c, 4*4*9, 1, 1, 0)

        self.level = level
        assert self.level in [4, 8, 16], "Bitch"

    def mask_conv(self, x):
        if self.level == 4:
            return self.maskconvx4(x)
        if self.level == 8:
            return self.maskconvx8(x)
        if self.level == 16:
            return self.maskconvx16(x)

    def upsample_flow(self, flow, mask):
        # 俺寻思俺懂了
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.level, self.level, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.level * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 4, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 4, self.level*H, self.level*W)

    def forward(self, x, scale):
        x = self.convblock(x) + x # 类似ResNet的f(x) + x
        tmp = self.flowconv(x)
        up_mask = self.mask_conv(x)
        flow_up = self.upsample_flow(tmp, up_mask)
        flow = F.interpolate(flow_up, scale_factor = scale, mode="bilinear", align_corners=False) * scale
        return flow
    
class IFUNet(nn.Module):
    def __init__(self):
        super(IFUNet, self).__init__()
        # block0通道数必须为128的整倍数
        self.fmap = FeatureNet(in_planes=17, out_planes=256)
        self.block0 = IFBlock(c=256, level=16)
        self.block1 = IFBlock(c=128, level=8)
        self.block2 = IFBlock(c=64, level=4)

    def forward(self, x, scale=1.0, timestep=0.5):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                x = torch.cat((img0, img1, timestep, warped_img0, warped_img1), 1)
                flowtmp = flow
                if scale != 1:
                    x = F.interpolate(x, scale_factor = scale, mode="bilinear", align_corners=False)
                    flowtmp = F.interpolate(flow, scale_factor = scale, mode="bilinear", align_corners=False) * scale
                x = torch.cat((x, flowtmp), 1)
                # 期待UNet能提取到特征，不再需要ensemble
                Fmap = self.fmap(x, level=i)
                flow_d = block[i](Fmap, scale=1. / scale)
                flow = flow + flow_d
            else:
                x = torch.cat((img0, img1, timestep), 1)
                if scale != 1:
                    x = F.interpolate(x, scale_factor = scale, mode="bilinear", align_corners=False)
                Fmap = self.fmap(x, level=i)
                flow = block[i](Fmap, scale=1. / scale)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        return flow, warped_img0, warped_img1
