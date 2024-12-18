import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data
from sklearn.mixture import GaussianMixture
from label_propagation import NonParaLP1
import scipy.sparse as sp

def accuracy(pred, ground_truth):
    y_hat = pred.max(1)[1]
    correct = (y_hat == ground_truth).nonzero().shape[0]
    acc = correct / ground_truth.shape[0]
    return acc * 100


def student_loss(s_logit, t_logit, return_t_logits=False, method="kl"):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if method == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif method == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(method)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

    
def construct_graph(node_logits, adj_logits, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    return data   
    

def random_walk_with_matrix(T, walk_length, start):
    current_node = start
    walk = [current_node]
    for _ in range(walk_length - 1):
        probabilities = F.softmax(T[current_node, :], dim=0)
        probabilities /= torch.sum(probabilities)
        next_node = torch.multinomial(probabilities, 1).item()
        walk.append(next_node)
        current_node = next_node
    return walk




def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    T = A * torch.pinverse(D)
    result_each_length = []
    
    for i in range(1, max_walk_length+1):    
        result_per_node = []
        for start in range(num_nodes):
            result_walk = random_walk_with_matrix(T, i, start)
            result_per_node.append(torch.tensor(result_walk).view(1,-1))
        result_each_length.append(torch.vstack(result_per_node))
    topo_emb = torch.hstack(result_each_length)
    return topo_emb    
def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

def calculate_loss_and_fit_gmm(logits, labels):

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(logits, labels)

    losses_np = losses.detach().cpu().numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(losses_np)
    
    probs = gmm.predict_proba(losses_np)

    small_mean_idx = np.argmin(gmm.means_)
    
    clean_samples = probs[:, small_mean_idx] > 0.5
    
    return clean_samples, losses

def normalize_adj_row(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    np.seterr(divide='ignore')
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    np.seterr(divide='warn')
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).tocoo()

def preprocess(args, data, soft_labels,eval_output1, high,low,device):
    """Preprocess the dataset."""
    num_nodes = data.x.shape[0]
    num_classes = data.y.max().item() + 1
    Y_all = torch.zeros(num_nodes, args.num_classes).to(device)
    Y = one_hot(soft_labels, args.num_classes).to(device)

    Y_all[data.train_idx] = Y[data.train_idx]
    Y_all[low]= eval_output1[low]
    # print(Y_all[low])
    # Y = soft_labels.to(device)
    # Y_all[data.train_idx] = Y[data.train_idx]
    #############
    edge_index_with_self_loops, _ = add_self_loops(data.edge_index)
    all_nodes = set(range(num_nodes))
    nodes_with_edges = set(edge_index_with_self_loops.view(-1).tolist())
    isolated_nodes = all_nodes - nodes_with_edges
    isolated_self_loops = torch.tensor([[i, i] for i in isolated_nodes], dtype=torch.long).t().to(device)
    edge_index_with_all_self_loops = torch.cat([edge_index_with_self_loops, isolated_self_loops], dim=1)
    A = to_dense_adj(edge_index_with_all_self_loops)[0].cpu()
    ######################
    # A = to_dense_adj(data.edge_index)[0].cpu()
    A = A.to(device)
    A1 = A.detach().clone()
    int_train_mask = data.train_idx.cpu().numpy().astype(int).reshape(-1, 1)
    M = np.matmul(int_train_mask, int_train_mask.T).astype(bool)
    A1 = A1.to('cpu') * M
    A1 = normalize_adj_row(A1.to('cpu'))
    A1 = torch.from_numpy(A1.todense()).to(device)
    A = A.to('cpu')
    A = normalize_adj_row(A)
    A = torch.from_numpy(A.todense()).to(device)
    for i in range(args.lp_prop):
        Y_all = (1 - args.Alpha) * torch.matmul(A1, Y_all) + args.Alpha * Y_all
    L_all = torch.argmax(Y_all, dim=1)
    return L_all, Y_all, A

def preprocess1(args, data, soft_labels,device):
    """Preprocess the dataset."""
    num_nodes = data.x.shape[0]
    num_classes = data.y.max().item() + 1
    Y_all = torch.zeros(num_nodes, args.num_classes).to(device)
    Y = one_hot(soft_labels, args.num_classes).to(device)
    Y_all[data.train_idx] = Y[data.train_idx]
    #############
    edge_index_with_self_loops, _ = add_self_loops(data.edge_index)
    all_nodes = set(range(num_nodes))
    nodes_with_edges = set(edge_index_with_self_loops.view(-1).tolist())
    isolated_nodes = all_nodes - nodes_with_edges
    isolated_self_loops = torch.tensor([[i, i] for i in isolated_nodes], dtype=torch.long).t().to(device)
    edge_index_with_all_self_loops = torch.cat([edge_index_with_self_loops, isolated_self_loops], dim=1)
    A = to_dense_adj(edge_index_with_all_self_loops)[0].cpu()
    ######################
    # A = to_dense_adj(data.edge_index)[0].cpu()
    A = A.to(device)
    A1 = A.detach().clone()
    int_train_mask = data.train_idx.cpu().numpy().astype(int).reshape(-1, 1)
    M = np.matmul(int_train_mask, int_train_mask.T).astype(bool)
    A1 = A1.to('cpu') * M
    A1 = normalize_adj_row(A1.to('cpu'))
    A1 = torch.from_numpy(A1.todense()).to(device)
    A = A.to('cpu')
    A = normalize_adj_row(A)
    A = torch.from_numpy(A.todense()).to(device)
    for i in range(args.lp_prop):
        Y_all = (1 - args.Alpha) * torch.matmul(A1, Y_all) + args.Alpha * Y_all
    L_all = torch.argmax(Y_all, dim=1)
    return L_all, Y_all, A


  
def dynamic_thresholds_by_class(losses, labels, w=1.0):  
    unique_classes = torch.unique(labels)  
    class_thresholds = {}  
    for clss in unique_classes:  
        class_losses = losses[labels == clss]  
        # print(torch.where(labels == clss))
        if class_losses.numel() == 0:  
            continue  
        mean_loss = torch.mean(class_losses) 

        std_loss = torch.std(class_losses)
        if len(class_losses) == 1:  
            std_loss = 0.0
        threshold = mean_loss + w * std_loss  
        class_thresholds[clss.item()] = threshold.item() 
  
    return class_thresholds

def get_clean_samples_by_class(losses, labels, thresholds):
    losses_np = losses.detach().cpu().numpy()  # Detach from the computation graph, move to CPU, and convert to numpy
    labels_np = labels.detach().cpu().numpy()  # Detach from the computation graph, move to CPU, and convert to numpy
    
    clean_samples = np.zeros_like(losses_np, dtype=bool)
    for clss, threshold in thresholds.items():
        class_mask = labels_np == clss
        clean_samples[class_mask] = losses_np[class_mask] < threshold
    return clean_samples

def get_clean_and_noisy_samples(subgraphs, local_models, global_model, args,device):
    global_model.eval()
    clean_samples= {client_id: [] for client_id in range(args.num_clients)}
    noisy_samples ={client_id: [] for client_id in range(args.num_clients)}
    sample_losses ={client_id: [] for client_id in range(args.num_clients)}
    sample1_losses ={client_id: [] for client_id in range(args.num_clients)}

    for client_id in range(args.num_clients):
        subgraph = subgraphs[client_id]
        x, edge_index, y = subgraph.x, subgraph.edge_index, subgraph.y
        train_idx = subgraph.train_idx
        logits = global_model.forward_full(x, edge_index)
        losses = nn.CrossEntropyLoss(reduction='none')(logits[train_idx], y[train_idx])
        sample_losses[client_id].append(losses)

    for client_id in range(args.num_clients):
        subgraph = subgraphs[client_id]
        y = subgraph.y[subgraph.train_idx]
        losses = sample_losses[client_id]

        if args.dynamic_loss:
            thresholds = dynamic_thresholds_by_class(losses[0], y, args.w1)
            clean_samples_stage_1_cid = get_clean_samples_by_class(losses[0], y, thresholds)
        else:
            logits = global_model.forward(subgraph)
            clean_samples_stage_1_cid, losses = calculate_loss_and_fit_gmm(logits[subgraph.train_idx], y)

        logits = global_model.forward(subgraph)
        eval_output1 = F.softmax(logits.detach(), dim=1)
        eval_output = torch.argmax(eval_output1, dim=1)

        if args.lp_pan:
            high_confidence_mask = torch.zeros_like(subgraph.train_idx, dtype=torch.bool)
            high_confidence_mask[subgraph.train_idx] = eval_output[subgraph.train_idx] == y
            low_confidence_mask = subgraph.train_idx & ~high_confidence_mask
            high_confidence_indices = torch.nonzero(high_confidence_mask).squeeze()
            low_confidence_indices = torch.nonzero(low_confidence_mask).squeeze()
            L_all, Y_all, A = preprocess(args, subgraph, eval_output, eval_output1,high_confidence_indices,low_confidence_indices,device)
        else:
            L_all, Y_all, A = preprocess1(args, subgraph, eval_output, device)
        
        loss2  = nn.CrossEntropyLoss(reduction='none')(Y_all[subgraph.train_idx], subgraph.y[subgraph.train_idx])

        if args.dynamic_loss1:
            thresholds1 = dynamic_thresholds_by_class(loss2, y, args.w2)
            clean_samples_stage_2_cid = get_clean_samples_by_class(loss2, y, thresholds1)
        else:
            clean_samples_stage_2_cid, losses = calculate_loss_and_fit_gmm(Y_all[subgraph.train_idx], y)

        if args.only_stage1:
            clean_samples_stage_2_cid = clean_samples_stage_1_cid
        else:
            clean_samples_stage_2_cid = clean_samples_stage_2_cid

        final_clean_samples = np.logical_and(clean_samples_stage_1_cid, clean_samples_stage_2_cid)
        final_noisy_samples = np.logical_not(final_clean_samples)

        clean_samples[client_id].append(final_clean_samples)
        noisy_samples[client_id].append(final_noisy_samples)

    return clean_samples, noisy_samples



def com_distillation_loss( t_logits, s_logits, edge_index, temp):
    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

    s_dist_neigh = F.log_softmax(s_logits[edge_index[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_index[1]] / temp, dim=-1)

    kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss
    
def con_loss( z1: torch.Tensor, z2: torch.Tensor, t):
    f = lambda x: torch.exp(x / t)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def filter_edges_by_mask(edge_index, mask):
    mask_indices = mask.nonzero(as_tuple=False).squeeze()
    edge_mask = (edge_index[0].unsqueeze(1) == mask_indices).any(dim=1) & (edge_index[1].unsqueeze(1) == mask_indices).any(dim=1)
    return edge_index[:, edge_mask]

def calculate_entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = torch.log(probabilities + 1e-9) 
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy.mean().item()  
