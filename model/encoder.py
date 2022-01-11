import torch
import torch.nn as nn
from model.backbone import p3da_resnet50 as backbone
from model.head import Classfier_Head, MLP_Head

class Encoder(nn.Module):

    def __init__(self, num_channels=128, mlp_layers=2, order=False, classfier=False, num_classes=400):
        super(Encoder, self).__init__()

        self.backbone = backbone()
        self.feature_size = self.backbone.feature_size
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.nce_inter_head = MLP_Head(mlp_layers, self.feature_size, num_channels)
        self.nce_intra_head = MLP_Head(mlp_layers, self.feature_size, num_channels)

        if order:
            self.order_head = MLP_Head(mlp_layers, self.feature_size, num_channels)
            self.order_classfier = Classfier_Head(num_channels * 3, 2)
        
        if classfier:
            self.classfier = Classfier_Head(self.feature_size, num_classes)

    
    def forward(self, x, mode='all'):

        """ mode = {'all', 'classfier_order', 'eval', 'classfier'} """
        
        # without backbone forward
        if mode == 'classfier_order':
            return self.forward_order_classfier(x)

        # with backbone forward
        out_list = []
        backbone_out = self.backbone(x, layer=5)
        if mode == 'all':
            out_list.extend(self.forward_nce_order(backbone_out))
        elif mode == 'eval':
            return backbone_out
        elif mode == 'classfier':
            return self.forward_classfier(backbone_out)
        
        return out_list if len(out_list) > 1 else out_list[0]

    def forward_nce_order(self, x):
        x = self.avg_pool(x).view(-1, self.feature_size)
        return nn.functional.normalize(self.nce_inter_head(x), p=2, dim=1), \
            nn.functional.normalize(self.nce_intra_head(x), p=2, dim=1), \
            nn.functional.normalize(self.order_head(x), p=2, dim=1)
    
    def forward_order_classfier(self, x):
        return self.order_classfier(x)

    def forward_classfier(self, x):
        x = self.avg_pool(x).view(-1, self.feature_size)
        return self.classfier(x)
    