import numpy as np
import torch
import model
import dataset
import os
from torch.utils.data import DataLoader
import argparse
import torch.distributed as dist
import train_loss
if __name__ =='__main__':
    parser = argparse.ArgumentParser("Unifying Global-Local Representations Transformer")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--batch_size_per_gpu", default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--base_lr", default=1e-3)
    parser.add_argument("--path", type=str)
    parser.add_argument("--pretrain", type=str)
    args = parser.parse_args()
    print("local_rank", args.local_rank)
    word_size = int(os.environ['WORLD_SIZE'])
    print("word size:", word_size)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    net = model.Model(args.pretrain, 384)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank])
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    Datasampler = torch.utils.data.distributed.DistributedSampler(Dataset, num_replicas=dist.get_world_size(), rank=args.local_rank, shuffle=True)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size_per_gpu, num_workers=args.batch_size_per_gpu * 2, collate_fn=dataset.my_collate_fn, sampler=Datasampler, drop_last=True)
    encoder_param=[]
    decoer_param=[]
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    for i in range(1, 200):
        if i==100 or i==150:
            for param_group in optimizer.param_groups:
                param_group['lr']= param_group['lr']*0.1
                print("Learning rate:", param_group['lr'])
        Datasampler.set_epoch(i)
        net.train()
        running_loss, running_loss0=0., 0.
        count=0
        for data in Dataloader:
            count+=1
            img, label = data['img'].cuda(args.local_rank), data['label'].cuda(args.local_rank)
            out = net(img)
            loss, loss0=train_loss.multi_bce(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            running_loss0+=loss0.item()
            if count%100==0 and args.local_rank==0:
                print("Epoch:{}, Iter:{}, loss:{:.5f}, loss0:{:.5f}".format(i, count, running_loss/count, running_loss0/count))
        if args.local_rank==0:
            if not os.path.exists("ckpt"):
                os.mkdir("ckpt")
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./ckpt/model_{}_loss_{:.5f}.pth".format(i, running_loss0/count))