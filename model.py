import torch
import torch.nn as nn
import torch.nn.functional as F
import vit


class decoder_stage(nn.Module):
    def __init__(self, infilter, outfilter):
        super(decoder_stage, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(infilter, outfilter, 1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(outfilter, outfilter, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(outfilter, outfilter, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        b,c,h,w = x.size()
        x= self.conv1(x)#F.interpolate(self.conv1(x), (h*4, w*4), mode='bilinear', align_corners=True)
        return x

class bridges4(nn.Module):
    def __init__(self, infilter, num):
        super(bridges4, self).__init__()
        outfilter=infilter*4
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges3(nn.Module):
    def __init__(self, infilter, num):
        super(bridges3, self).__init__()
        outfilter=infilter*2
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.UpsamplingBilinear2d(scale_factor=2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges2(nn.Module):
    def __init__(self, infilter, num):
        super(bridges2, self).__init__()
        outfilter = infilter*4
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter), nn.ReLU(inplace=True),
                nn.PixelShuffle(4),
                nn.UpsamplingBilinear2d(scale_factor=2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges1(nn.Module):
    def __init__(self, infilter, num):
        super(bridges1, self).__init__()
        outfilter = infilter*2
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False), nn.BatchNorm2d(outfilter), nn.ReLU(inplace=True), nn.PixelShuffle(4)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(F.interpolate(self.modulelist[i](features[i]), (192, 192), mode='bilinear'))
        return feature


class outs(nn.Module):
    def __init__(self, infilter, num, first, scale_factor=4):
        super(outs, self).__init__()
        self.modulelist = nn.ModuleList()
        self.num = num
        if first:
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter//2, infilter//2, 1, bias=False),
                                                 nn.BatchNorm2d(infilter//2),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter//2, 1, 1),
                                                 nn.Sigmoid()))
        else:
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter, infilter, 1, bias=False),
                                                 nn.BatchNorm2d(infilter),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter, 1, 1),
                                                 nn.Sigmoid()))
        for i in range(num-1):
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter, infilter, 1, bias=False),
                                                 nn.BatchNorm2d(infilter),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter, 1, 1),
                                                 nn.Sigmoid()))

    def forward(self, features):
        assert self.num == len(features)
        feature = []
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature


class Model(nn.Module):
    def __init__(self, ckpt, imgsize=384):
        super(Model, self).__init__()
        self.encoder = vit.deit_base_distilled_patch16_384()
        self.imgsize=imgsize
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location='cpu')
            msg = self.encoder.load_state_dict(ckpt["model"], strict=False)
            print(msg)
        self.out1 = outs(192, 4, first=False, scale_factor=2)
        self.out2 = outs(384, 4, first=True, scale_factor=4)
        self.out3 = outs(768, 4, first=True, scale_factor=8)
        self.bridge3 = bridges4(768, 4)
        self.bridge2 = bridges3(768, 4)
        self.bridge1 = bridges2(768, 4)
        self.decoder12 = decoder_stage(768, 768)
        self.decoder11 = decoder_stage(768 * 2, 768)
        self.decoder10 = decoder_stage(768 * 2, 768)
        self.decoder9 = decoder_stage(768 * 2, 384)
        self.decoder8 = decoder_stage(384 * 2, 384)
        self.decoder7 = decoder_stage(384 * 2, 384)
        self.decoder6 = decoder_stage(384 * 2, 384)
        self.decoder5 = decoder_stage(384 * 2, 192)
        self.decoder4 = decoder_stage(192 * 2, 192)
        self.decoder3 = decoder_stage(192 * 2, 192)
        self.decoder2 = decoder_stage(192*2, 192)
        self.decoder1 = decoder_stage(192*2, 192)

    def decoder(self, feature):
        feature12 = self.decoder12(feature[-1])
        feature11 = self.decoder11(torch.cat((feature12, feature[-2]), 1))
        feature10 = self.decoder10(torch.cat((feature11, feature[-3]), 1))
        feature9 = self.decoder9(torch.cat((feature10, feature[-4]), 1))
        feature8 = self.decoder8(torch.cat((F.interpolate(feature9, (96, 96), mode='bilinear'), feature[-5]), 1))
        feature7 = self.decoder7(torch.cat((feature8, feature[-6]), 1))
        feature6 = self.decoder6(torch.cat((feature7, feature[-7]), 1))
        feature5 = self.decoder5(torch.cat((feature6, feature[-8]), 1))
        feature4 = self.decoder4(torch.cat((F.interpolate(feature5, (192, 192), mode='bilinear'), feature[-9]), 1))
        feature3 = self.decoder3(torch.cat((feature4, feature[-10]), 1))
        feature2 = self.decoder2(torch.cat((feature3, feature[-11]), 1))
        feature1 = self.decoder1(torch.cat((feature2, feature[-12]), 1))
        return [feature1, feature2, feature3, feature4, feature5, feature6,
                feature7, feature8, feature9, feature10, feature11, feature12]
    def forward(self, img):
        # B Seq
        B, C, H, W = img.size()
        x = self.encoder(img)
        feature = []  #3, 6, 9, 12
        for x0 in x:
            feature.append(x0[:, 2:, :].permute(0, 2, 1).view(B, 768, int(self.imgsize/16), int(self.imgsize/16)).contiguous())
        feature = self.bridge1(feature[:4])+self.bridge2(feature[4:8])+self.bridge3(feature[8:])
        feature = self.decoder(feature)
        return self.out1(feature[:4]) + self.out2(feature[4:8])+self.out3(feature[8:])
