import torch
import torch.nn as nn
import torch.nn.functional as F

def prediction(g, f, loader):
    g.eval()
    f.eval()
    P, F = [], []
    with torch.no_grad():
        for data in loader:
            x = data[0][0].float().cuda()
            feat = g(x)
            _, out = f(feat)
            F.append(feat)
            P.append(out)
    g.train()
    f.train()
    return torch.vstack(P), torch.vstack(F)

def prediction_with_label(loader, model):
    g.eval()
    f.eval()
    P, F, Y = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.cuda().float()
            F.append(model.get_features(x))
            P.append(model.get_predictions(F[-1]))
            Y.append(y)
    g.train()
    f.train()
    return torch.vstack(P), torch.vstack(F), torch.hstack(Y)

class ProtoClassifier(nn.Module):
    def __init__(self, size):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = None
        self.size = size
    def init(self, g, f, t_loader):
        t_pred, t_feat = prediction(g, f, t_loader)
        label = t_pred.argmax(dim=1)
        center = torch.nan_to_num(torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.size)]))
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            old_center = torch.vstack([t_feat[self.label == i].mean(dim=0) for i in range(self.size)])
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)
    def ideal_init(self, model, t_loader):
        _, t_feat, label = prediction_with_label(t_loader, model)
        center = torch.nan_to_num(torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.size)]))
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            old_center = torch.vstack([t_feat[self.label == i].mean(dim=0) for i in range(self.size)])
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist*T, dim=1)
