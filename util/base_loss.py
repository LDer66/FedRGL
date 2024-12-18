import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
increment = 0.5/300
def custom_loss_fn(model, embedding, logits, label, mask, embeddingg, round_id,  device,args):
    task_loss = nn.CrossEntropyLoss()(logits[mask], label[mask])
    
    if round_id == 0:
        return task_loss
    else:
        sim_global = torch.cosine_similarity(embedding, embeddingg, dim=-1).view(-1, 1)
        sim_prev = torch.cosine_similarity(embedding, model.prev_local_embedding, dim=-1).view(-1, 1)
        logits = torch.cat((sim_global, sim_prev), dim=1) / args.temperature
        lbls = torch.zeros(embedding.size(0)).to(device).long()
        contrastive_loss = nn.CrossEntropyLoss()(logits, lbls)
        moon_loss = args.moon_mu * contrastive_loss
        return task_loss + args.moon_mu*moon_loss

def criterion_1( y_1, y_2, t, co_lambda=0.1, epoch=-1):
        # loss_pick_1 = F.cross_entropy(y_1, t, reduce=False)
        # loss_pick_2 = F.cross_entropy(y_2, t, reduce=False)

        loss_pick_1 =  F.cross_entropy(y_1, t, reduction='none')
        loss_pick_2 =  F.cross_entropy(y_2, t, reduction='none')
        loss_pick = loss_pick_1  + loss_pick_2

        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = increment*epoch
        remember_rate = 1 - forget_rate
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted<mean_v)[0]
      
        remember_rate_small = idx_small.shape[0]/t.shape[0]
       
        remember_rate = max(remember_rate,remember_rate_small)
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
    
        loss_clean = torch.sum(loss_pick[ind_update])/y_1.shape[0]
        ind_all = torch.arange(1, t.shape[0]).long()
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(ind_update.detach().cpu().numpy())))
        p_1 = F.softmax(y_1,dim=-1)
        p_2 = F.softmax(y_2,dim=-1)
        
        filter_condition = ((y_1.max(dim=1)[1][ind_update_1] != t[ind_update_1]) &
                            (y_1.max(dim=1)[1][ind_update_1] == y_2.max(dim=1)[1][ind_update_1]) &
                            (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (1-(1-min(0.5,1/y_1.shape[0]))*epoch/300)))
        dc_idx = ind_update_1[filter_condition]
        
        adpative_weight = (p_1.max(dim=1)[0][dc_idx]*p_2.max(dim=1)[0][dc_idx])**(0.5-0.5*epoch/300)
        loss_dc = adpative_weight*(F.cross_entropy(y_1[dc_idx],y_1.max(dim=1)[1][dc_idx], reduction='none')+ \
                                   F.cross_entropy(y_2[dc_idx], y_1.max(dim=1)[1][dc_idx], reduction='none'))
        loss_dc = loss_dc.sum()/y_1.shape[0]
    
        remain_idx = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(dc_idx.detach().cpu().numpy())))
        
        loss1 = torch.sum(loss_pick[remain_idx])/y_1.shape[0]
        # decay_w = 5e-4
        decay_w = 0.01

        inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()

        return loss_clean + loss_dc+decay_w*loss1+co_lambda*inter_view_loss

def intra_reg( out1, out2, edge_index):
    shape = out1.shape[0]
    pred_u1 = out1[edge_index[0]]
    pred_v1 = out1[edge_index[1]]
    pred_u2 = out2[edge_index[0]]
    pred_v2 = out2[edge_index[1]]        
    loss = kl_loss_compute(pred_u1, pred_v1.detach()) + kl_loss_compute(pred_u2, pred_v2.detach())
    loss = loss / shape
    return loss

def kl_loss_compute( pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
    
def filter_edges_by_mask(edge_index,train_mask):
    # train_nodes = torch.where(train_mask)[0]
    mask = (train_mask[edge_index[0]] & train_mask[edge_index[1]])
    filtered_edge_index = edge_index[:, mask]
    # # 获取 edge_index 中的节点
    # node1 = edge_index[0, :]
    # node2 = edge_index[1, :]

    # # 创建一个布尔掩码，表示边的两个节点都在 train_mask 中
    # mask = train_mask[node1] & train_mask[node2]

    # # 根据布尔掩码筛选边索引
    # edge_index_train = edge_index[:, mask]
    
    return filtered_edge_index

eps = 1e-7

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    

def get_output(subgraphs, net, args, device,latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        # subgraphs = subgraphs.to(device)

        outputs = net.forward(subgraphs)
        loss = criterion(outputs[subgraphs.train_idx],subgraphs.y[subgraphs.train_idx])

    return outputs,loss