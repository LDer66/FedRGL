import random

import numpy as np
import torch
import torch.nn as nn
from ctypes import c_int
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from util.fgl_dataset import FGLDataset
import scipy.sparse as sp
import os.path as osp
import numpy.ctypeslib as ctl


def load_dataset(args):
    if args.dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "CS",
        "Physics",
        "NELL",
        "Wiki",
    ]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.2,
            val=0.4,
            test=0.4,
            part_delta=args.part_delta,
        )
    elif args.dataset in ["ogbn-arxiv", "Flickr"]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.6,
            val=0.2,
            test=0.2,
        )
    elif args.dataset in ["Reddit"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.8,
            val=0.1,
            test=0.1,
        )
    elif args.dataset in ["ogbn-products"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.1,
            val=0.05,
            test=0.85,
        )
    return dataset


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def lid_term(X, batch, k=20):
        eps = 1e-6
        X = np.asarray(X, dtype=np.float32)

        batch = np.asarray(batch, dtype=np.float32)
        f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
        distances = cdist(X, batch)

        # get the closest k neighbours
        sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
        m, n = sort_indices.shape
        idx = np.ogrid[:m, :n]
        idx[1] = sort_indices
        # sorted matrix
        distances_ = distances[tuple(idx)]
        lids = np.apply_along_axis(f, axis=1, arr=distances_)
        return lids

def view_grad(input, model_list=None):
    if model_list is not None:
        make_dot(input, dict([model.named_parameters() for model in model_list])).view()
    else:
        make_dot(input).view()

def sparseTensor_to_coomatrix(edge_idx, num_nodes):
    if edge_idx.shape == torch.Size([0]):
        adj = coo_matrix((num_nodes, num_nodes), dtype=np.int)
    else:
        row = edge_idx[0].cpu().numpy()
        col = edge_idx[1].cpu().numpy()
        data = np.ones(edge_idx.shape[1])
        adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.int)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def homo_adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized

def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    ctl_lib = ctl.load_library("libmatmul.so", dir_path)
    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None
    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape
    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)
    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

