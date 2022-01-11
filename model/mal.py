import torch
import torch.nn as nn
import torch.nn.functional as F

class MAL(nn.Module):
    def __init__(self):
        super(MAL, self).__init__()
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.softmax = nn.Softmax(dim=1)

    def alignment(self, gradient, xconv, target):

        bsz = gradient.size(0)
        weight = self.avgpool(gradient)

        weight_predict = weight * gradient

        predict = self.relu(weight_predict.sum(1))

        # for spatial-temporal alignment
        sim_spa_temp = self.forward_spatial_temporal(predict, target, bsz)

        # for temporal alignment
        sim_tempo = self.forward_temporal(predict, target, bsz)

        # for spa alignment
        sim_spa = self.forward_spatial(predict, target, bsz)

        return sim_spa_temp, sim_spa, sim_tempo

    def forward_spatial_temporal(self, predict, target, bsz):

        target_score = target.reshape(bsz, 4, -1).sum(-1)
        target_att = self.softmax(target_score)
        pre_norm = F.normalize(predict.reshape(bsz, 4, -1), dim=-1)
        target_norm = F.normalize(target.reshape(bsz, 4, -1), dim=-1)
        sim = (pre_norm * target_norm).sum(-1)
        sim_att = (sim * target_att).sum(-1)
        return sim_att

    def forward_temporal(self, predict, target, bsz):
        pre_score = predict.reshape(bsz, 4, -1).sum(-1)
        motion_score = target.reshape(bsz, 4, -1).sum(-1)
        pre_norm = F.normalize(pre_score, dim=-1)
        target_norm = F.normalize(motion_score, dim=-1)
        sim = (pre_norm * target_norm).sum(-1)
        return sim

    def forward_spatial(self,  predict, target, bsz):
        pre_norm = F.normalize(predict.reshape(bsz, 4, -1).mean(1), dim=-1)
        target_norm = F.normalize(target.reshape(bsz, 4, -1).mean(1), dim=-1)
        sim = (pre_norm * target_norm).sum(-1)
        return sim

    def forward(self, gradient, xconv, target):

        sim_spa_temp, sim_spa, sim_tempo = self.alignment(gradient, xconv, target)

        # cosine similarity equal to l2-normalized mse
        loss_spa_temp = (1 - sim_spa_temp).mean()
        loss_tempo = (1 - sim_tempo).mean()
        loss_spa = (1 - sim_spa).mean()

        loss_mal = loss_spa_temp + loss_spa + loss_tempo

        return loss_mal