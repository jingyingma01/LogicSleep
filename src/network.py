import torch
import torch.nn as nn
from torchcrf import CRF
import utils.util as util
from utils.ops import GraphUnet, Initializer, norm_g

class GNet(nn.Module):
    def __init__(self, config):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, config.act_n)()
        self.c_act = getattr(nn, config.act_c)()
        self.num_patch = config.num_patch
        self.patch_width = config.num_node
        self.g_unet = GraphUnet(config)
        self.outl = nn.Linear(3000, 1)
        self.out_drop = nn.Dropout(p = config.drop_c)
        self.crf = CRF(num_tags=config.num_class, batch_first=True).to(util.set_device(config))
        Initializer.weights_init(self)

    def forward(self, gs, hs):
        hs, salient_spacial, salient_node, ts_trasaction = self.embed(gs, hs)
        raw_logits, embedding = self.classify(hs)
        return raw_logits, embedding, salient_spacial, salient_node, ts_trasaction

    def embed(self, gs, hs):
        gs = norm_g(gs)
        hs, salient_spacial, salient_node, ts_trasaction = self.g_unet(gs, hs)
        return hs, salient_spacial, salient_node, ts_trasaction

    def classify(self, h):
        h = self.out_drop(h)
        patches = [h[:, :, i * self.patch_width : (i + 1) * self.patch_width]
                   for i in range(self.num_patch)]
        embedding = torch.cat(patches, dim = 3)
        h = embedding.mean(dim = 2, keepdim = False)
        h = torch.relu(h)
        h = self.outl(h).squeeze()
        return h, embedding

    @torch.no_grad()
    def crf_decode(self, emissions, mask):
        return self.crf.decode(emissions, mask = mask)

    def crf_nll(self, emissions, tags, mask, reduction = 'mean'):
        return - self.crf(emissions, tags, mask = mask, reduction = reduction)