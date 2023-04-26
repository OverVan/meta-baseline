import torch
from torch import nn

from .model import register, make
from utils import cosine_similarity


@register("meta_baseline")
class PaperModel(nn.Module):
    def __init__(self, encoder, tao=10.):
        super(PaperModel, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        
    def forward(self, x, n, k, q):
        x = self.encoder(x).reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        proto = support.mean(1)
        sim = cosine_similarity(query, proto)
        return self.tao * sim