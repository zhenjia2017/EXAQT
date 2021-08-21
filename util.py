import yaml
import torch
import os
import json
import numpy as np
import pickle
from torch.autograd import Variable
from operator import itemgetter, attrgetter
import warnings

def get_config(config_path):
    print ("\nconfig_path:", config_path)
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def save_model(the_model, path):
    if os.path.exists(path):
        path = path + '_copy'
    print("saving model to ...", path)
    torch.save(the_model, path)

def load_model(path):
    if not os.path.exists(path):
        assert False, 'cannot find model: ' + path
    return torch.load(path)

def load_date(filename):
    dates = []
    with open(filename) as f_in:
        for line in f_in:
            date_range = dict()
            date = line.strip().replace("T00: 00:00Z","")
            if date.startswith("-"):
                year = date.split("-")[1]
                month = date.split("-")[2]
                day = date.split("-")[3]
            else:
                year = date.split("-")[0]
                month = date.split("-")[1]
                day = date.split("-")[2]
            date_range['year'] = int(year)
            date_range['month'] = int(month)
            date_range['day'] = int(day)
            dates.append(date_range)

        specs = (('year', False), ('month', False), ('day', False))
        for key, reverse in reversed(specs):
            dates.sort(key=itemgetter(key), reverse=reverse)

    return dates

def load_map(filename):
    word2id = pickle.load(open(filename, 'rb'))
    return word2id

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            #word = line.strip().decode('UTF-8')
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)

def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        f_pred.write(json.dumps({'dist': output_dist, 'answers':answers, 'seeds': data_loader.data[data_id]['entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

class LeftMMFixed(torch.autograd.Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """

    @staticmethod
    def forward(ctx, sparse_weights, x):
        ctx.save_for_backward(sparse_weights, x)
        output = torch.mm(sparse_weights, x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_weights, x = ctx.saved_variables
        return None, torch.mm(sparse_weights.t(), grad_output)

def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    lookup = Y[I[0, :], I[2, :], :]
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.cuda.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))

    prod_op = LeftMMFixed.apply
    prod = prod_op(S, lookup)
    warnings.filterwarnings("ignore")
    return prod.view(B, M, K)

