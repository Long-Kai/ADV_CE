import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import torch.nn as nn

class FC_Classifier(nn.Module):
    def __init__(self, feat_dim):

        super(FC_Classifier, self).__init__()

        dim_input = feat_dim
        dim_output = 5

        self.net = nn.Linear(dim_input, dim_output)
        self.net.bias.data = torch.zeros(5)

    def forward(self, x, params=None):
        if params is not None:
            (weight, bias) = params["weight"], params["bias"]
            out = F.linear(input=x, weight=weight, bias=bias)
        else:
            out = self.net(x)

        return out

    def init(self, weight):
        self.net.weight.data = weight.detach().clone()
        self.net.bias.data = torch.zeros_like(self.net.bias)


def unit_normalize(x):
    x_sum = x.sum(dim=1, keepdims=True)
    x_sum[x_sum==0] = 1

    return x / x_sum

def set_not_top_k_to_zero(x, k=2):
    n, d = x.shape
    x[(torch.arange(n).unsqueeze(1), x.topk(d-k, largest=False).indices)] = 0
    return x


def random_walk(start, nodes, nnk, step, tau):
    """
    nnk:  num of activate nearest nodes for next step
    step: num of step to walk between nodes
    tau:  temperature for calculating the transition probablity based on distance
    """
    n = nodes.shape[0]

    d_s_n = (torch.cdist(start.unsqueeze(0), nodes.unsqueeze(0), p=2.0).squeeze(0)) ** 2 * tau
    # d_n_s = d_s_n.clone().T
    d_n_n = (torch.cdist(nodes.unsqueeze(0), nodes.unsqueeze(0), p=2.0).squeeze(0)) ** 2 * tau
    d_n_n[np.arange(n), np.arange(n)] = np.inf

    p_s_n = torch.softmax(-d_s_n, dim=1)
    p_n_n = torch.softmax(-d_n_n, dim=1)
    # p_n_s = torch.softmax(-d_n_s, dim=1)

    p_s_n = set_not_top_k_to_zero(p_s_n, nnk)
    p_s_n = unit_normalize(p_s_n)

    p_n_n = set_not_top_k_to_zero(p_n_n, nnk)
    p_n_n = unit_normalize(p_n_n)

    trans = p_s_n
    for i in range(step-1):
        trans = trans @ p_n_n
        trans = unit_normalize(trans)

    return trans


def vd_rw(support_data, query_data, cfg):
    """
    random walk-based vicinal distribution
    """
    nnk = cfg['nn_k']
    tau = cfg['tau']
    step = cfg['rw_step']

    for i in range(step):
        weight = random_walk(support_data, query_data, nnk=nnk, step=step, tau=tau)
        if i == 0:
            weight_mean = weight @ query_data
            weight_cov_diag = (weight.unsqueeze(-1) * ((query_data.unsqueeze(0)-weight_mean.unsqueeze(1))**2)).sum(dim=1)
        else:
            weight_mean = weight_mean + weight @ query_data
            weight_cov_diag = weight_cov_diag + (weight.unsqueeze(-1) * ((query_data.unsqueeze(0)-weight_mean.unsqueeze(1))**2)).sum(dim=1)

    vd_mean = (support_data + weight_mean) / (step + 1)
    vd_cov_diag = ((support_data - weight_mean) ** 2 + weight_cov_diag) / (step + 1)

    return vd_mean, vd_cov_diag


def vicinal_loss_fn(input, target, sigma_diag, sigma_bias, weight):
    """
    input: logist, bxc
    target: label, b
    sigma_diag: Sigma matrix (given as diag vec), b x feat_dim
    sigma_bias: a bias term added to the Sigma matrix, b x 1
    """
    z = input

    nSample = z.size(0)
    sigma_wij = sigma_bias.reshape(-1, 1, 1) * (weight.sum(1, keepdim=True).mm(weight.sum(1, keepdim=True).T).unsqueeze(0)) \
                + (sigma_diag.unsqueeze(1) * weight.unsqueeze(0)).matmul((weight.T).unsqueeze(0))


    loss = F.cross_entropy(input=z, target=target, reduction='mean')
    p = F.softmax(z, dim=1)

    sigma_w = torch.diagonal(sigma_wij, offset=0, dim1=-2, dim2=-1)
    loss = loss + ((p * sigma_w).sum() - p.unsqueeze(1).bmm(sigma_wij).bmm(p.unsqueeze(2)).sum())/nSample

    return loss

def adv_ce(support_data, support_label, query_data, query_label, cfg):
    device = cfg['device']
    n_ways = cfg['n_ways']
    feat_dim = cfg['feat_dim']

    n_step = cfg['n_step']
    inner_lr = cfg['inner_lr']
    weight_decay = cfg['weight_decay']
    is_proto_init = cfg['proto_init']

    classifier = FC_Classifier(feat_dim).to(device)

    support_data = torch.tensor(support_data).to(device)
    query_data = torch.tensor(query_data).to(device)
    support_label = torch.tensor(support_label).to(device)
    query_label = torch.tensor(query_label).to(device)

    support_data, sigma_diag = vd_rw(support_data, query_data, cfg)
    sigma_bias = cfg['sigma_bias'] * torch.ones(support_data.shape[0]).to(device).float()

    if is_proto_init:
        proto = torch.zeros(n_ways, feat_dim).to(device).float()
        for j in range(n_ways):
            proto[j] = (support_data[support_label == j].mean(dim=0))
        classifier.init(proto)
        fast_weight = OrderedDict(classifier.net.named_parameters())
    else:
        fast_weight = OrderedDict(classifier.net.named_parameters())

    for k in range(n_step):
        preds_supp = classifier.forward(support_data, params=fast_weight)

        loss_supp = vicinal_loss_fn(input=preds_supp, target=support_label, sigma_diag=sigma_diag, sigma_bias=sigma_bias, weight=fast_weight['weight'])

        grads = torch.autograd.grad(loss_supp, fast_weight.values(), create_graph=True, allow_unused=True)

        fast_weight = OrderedDict(
            (name, param - inner_lr * grad - inner_lr * weight_decay * param) for ((name, param), grad) in zip(fast_weight.items(), grads)
        )

    preds_query = classifier.forward(query_data, fast_weight)

    _, predicted = torch.max(preds_query.data, 1)
    acc = predicted.float().eq(query_label.data.float()).cpu().float().numpy()

    return np.mean(acc)


def adv_ce_tim(support_data, support_label, query_data, query_label, cfg, tim_para):
    device = cfg['device']
    n_ways = cfg['n_ways']
    feat_dim = cfg['feat_dim']

    n_step = cfg['n_step']
    inner_lr = cfg['inner_lr']
    weight_decay = cfg['weight_decay']
    is_proto_init = cfg['proto_init']

    classifier = FC_Classifier(feat_dim).to(device)

    support_data = torch.tensor(support_data).to(device)
    query_data = torch.tensor(query_data).to(device)
    support_label = torch.tensor(support_label).to(device)
    query_label = torch.tensor(query_label).to(device)

    support_data, sigma_diag = vd_rw(support_data, query_data, cfg)
    sigma_bias = cfg['sigma_bias'] * torch.ones(support_data.shape[0]).to(device).float()

    if is_proto_init:
        proto = torch.zeros(n_ways, feat_dim).to(device).float()
        for j in range(n_ways):
            proto[j] = (support_data[support_label == j].mean(dim=0))
        classifier.init(proto)
        fast_weight = OrderedDict(classifier.net.named_parameters())
    else:
        fast_weight = OrderedDict(classifier.net.named_parameters())

    for k in range(n_step):
        preds_supp = classifier.forward(support_data, params=fast_weight)

        loss_supp = vicinal_loss_fn(input=preds_supp, target=support_label, sigma_diag=sigma_diag, sigma_bias=sigma_bias, weight=fast_weight['weight'])

        logits_q = classifier.forward(query_data, params=fast_weight)
        q_probs = logits_q.softmax(1)
        q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(1).mean(0)
        q_ent = - (q_probs.mean(0) * torch.log(q_probs.mean(0))).sum(0)
        loss_tim = (- 10.0 * q_ent + q_cond_ent) * tim_para

        loss = loss_supp + loss_tim

        grads = torch.autograd.grad(loss, fast_weight.values(), create_graph=True, allow_unused=True)

        fast_weight = OrderedDict(
            (name, param - inner_lr * grad - inner_lr * weight_decay * param) for ((name, param), grad) in zip(fast_weight.items(), grads)
        )

    preds_query = classifier.forward(query_data, fast_weight)

    _, predicted = torch.max(preds_query.data, 1)
    acc = predicted.float().eq(query_label.data.float()).cpu().float().numpy()

    return np.mean(acc)


def adv_svm(support_data, support_label, query_data, query_label, cfg, gamma=1.0, maxIter=3):
    from svm_vrm import VRM_SVM_CS
    device = cfg['device']
    n_ways = cfg['n_ways']
    n_shot = cfg['n_shot']

    support_data = torch.tensor(support_data).to(device)
    query_data = torch.tensor(query_data).to(device)
    support_label = torch.tensor(support_label).to(device)
    query_label = torch.tensor(query_label).to(device)

    support_data, sigma_diag = vd_rw(support_data, query_data, cfg)
    sigma = sigma_diag * gamma

    support_data = support_data.unsqueeze(0)
    support_label = support_label.unsqueeze(0)
    query_data = query_data.unsqueeze(0)
    query_label = query_label.unsqueeze(0)
    sigma = sigma.unsqueeze(0)

    preds_query = VRM_SVM_CS(support_data, sigma, support_label, query_data, n_ways, n_shot, gamma=gamma, maxIter=maxIter)

    preds_query = preds_query.squeeze(0)

    _, predicted = torch.max(preds_query.data, 1)
    acc = predicted.float().eq(query_label.data.float()).cpu().float().numpy()

    return np.mean(acc)




