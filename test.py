import torch
import model
import dataset
import os
from torch.utils.data import DataLoader
import train_loss
import numpy as np
import torch.nn.functional as F
from imageio import imwrite

if __name__ =='__main__':
    batch_size = 1
    net = model.Model(None, imgsize=384).cuda()
    ckpt=['ckpt.pth']
    Dirs=["/path/to/DUT-OMRON",
          "/path/to/ECSSD",
          "/path/to/HKU-IS",
          "/path/to/DUTS-TE",
          "/path/to/PASCAL-S",
          "/path/to/SOD"]
    for m in ckpt:
        print(m)
        pretrained_dict = torch.load("./ckpt/"+m)
        net_dict = net.state_dict()
        pretrained_dict={k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict }
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists("results"):
                os.mkdir("results")
            if not os.path.exists(os.path.join("results", Dir.split("/")[-1])):
                os.mkdir(os.path.join("results", Dir.split("/")[-1]))
            Dataset = dataset.TestDataset(Dir, 384)
            Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            count=0
            for data in Dataloader:
                count+=1
                img, label = data['img'].cuda(), data['label'].cuda()
                name = data['name'][0].split("/")[-1]
                with torch.no_grad():
                    out = net(img)[0]
                B,C,H,W = label.size()
                o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
                o = (o*255).astype(np.uint8)
                imwrite("./results/"+Dir.split("/")[-1]+"/"+name, o)

