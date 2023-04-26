import os
import torch
import random
import scipy.stats
import numpy as np
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def log(content, path="log/train.txt"):
    print(content)
    if not os.path.exists("./log"):
        os.mkdir("log")
    with open(path, "a", encoding="UTF-8") as file:
        print(content, file=file)
        

def cosine_similarity(x1, x2):
    assert x1.dim() == 2 and x2.dim() == 2, "2-d needed"
    assert x1.shape[1] == x2.shape[1], "feature dim should be equal"
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    return torch.mm(x1, x2.T)


def make_ep_label(n, q):
    label = torch.arange(n).unsqueeze(1).expand(n, q).reshape(-1)
    return label.cuda()


def make_optimizer(name, params, lr, weight_decay):
    optimizers = {
        "SGD": SGD
    }
    return optimizers[name](params, lr, momentum=0.9, weight_decay=weight_decay)


def make_lr_scheduler(name, optimizer, milestones):
    schedulers = {
        "MultiStepLR": MultiStepLR
    }
    return schedulers[name](optimizer, milestones=milestones)


def fix_seed(seed=0):
    random.seed(seed)
    # 禁止hash随机化，使得实验可复现
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 针对多GPU
    torch.cuda.manual_seed_all(seed)
