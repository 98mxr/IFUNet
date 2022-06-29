import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.warplayer import warp
from model.IFUNet import IFUNet
from model.rrdb import RRDBNet
from model.ResynNet import ResynNet

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFUNet()
        self.fusionnet = RRDBNet()
        self.refinenet = ResynNet()
        self.device()
        self.version = 3.9

    def eval(self):
        self.flownet.eval()
        self.fusionnet.eval()
        self.refinenet.eval()

    def device(self):
        self.flownet.to(device)
        self.fusionnet.to(device)
        self.refinenet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            self.flownet.load_state_dict(
                convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/fusionnet.pkl'.format(path), map_location=device)))
            self.refinenet.load_state_dict(
                convert(torch.load('{}/refinenet.pkl'.format(path), map_location=device)))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        flow, warped_img0, warped_img1 = self.flownet(imgs, scale, timestep)
        mask = self.fusionnet(img0, img1, warped_img0, warped_img1, flow)
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        merged, _ = self.refinenet(imgs, deg=merged, scale=[4, 2, 1])
        return merged