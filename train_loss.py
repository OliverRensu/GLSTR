import torch.nn as nn
bce = nn.BCELoss(reduction='mean')

def multi_bce(pred, gt):
    loss0= bce(pred[0], gt)
    loss=0.
    for i in range(1, len(pred)):
        loss+=bce(pred[i], gt)*(1-i/20)
    return loss+loss0, loss0