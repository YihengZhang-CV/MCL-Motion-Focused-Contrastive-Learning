import torch
import math
from torch import nn


class MemoryMCL(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.07, temperature_intra=1.0, multi_clip=True):
        super(MemoryMCL, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.temperature_intra = temperature_intra
        self.index = 0
        self.multi_clip = multi_clip
        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        
    def forward(self, q, k_sf, k_df1=None, k_df2=None, k_all_sf=None, k_all_df1=None, k_all_df2=None, inter=True):
        l_pos_sf = (q * k_sf.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        if inter:
            l_neg = torch.mm(q, self.memory.clone().detach().t())
            if self.multi_clip:
                l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
                l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
                out = torch.cat((torch.cat((l_pos_sf, l_pos_df1, l_pos_df2), dim=0), l_neg.repeat(3, 1)), dim=1)
            else:
                out = torch.cat((l_pos_sf, l_neg), dim=1)

            out = torch.div(out, self.temperature).contiguous()
            with torch.no_grad(): # update memory
                k_all = torch.cat((k_all_sf, k_all_df1, k_all_df2), dim=0) if self.multi_clip else k_all_sf
                all_size = k_all.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
                self.memory.index_copy_(0, out_ids, k_all)
                self.index = (self.index + all_size) % self.queue_size
                return out, l_pos_sf
        else:
            # out intra-frame similarity
            l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            out = torch.div(torch.cat((l_pos_sf.repeat(2, 1), torch.cat((l_pos_df1, l_pos_df2), dim=0)), dim=-1), self.temperature_intra).contiguous()
            return out


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        label = torch.zeros([x.shape[0]]).long().to(x.device)
        return self.criterion(x, label)
