import torch
import torch.nn as nn

class MLP_Head(nn.Module):
    def __init__(self, layers, input_channel, output_channel):
        super(MLP_Head, self).__init__()
        linear_list = []
        for i in range(layers):
            if i == layers - 1:
                linear_list.append(nn.Linear(input_channel, output_channel))
            else:
                linear_list.append(nn.Linear(input_channel, input_channel))
                linear_list.append(nn.ReLU())

        self.head = nn.Sequential(*linear_list)
        
    
    def forward(self, input):
        return self.head(input)

class Classfier_Head(nn.Module):
    def __init__(self, input_channel, num_classes, dropout_ratio=0.0, add_norm=True):
        super(Classfier_Head, self).__init__()
        self.fc = nn.Linear(input_channel, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.add_norm = add_norm

    def forward(self, input):
        if self.add_norm:
            input = torch.nn.functional.normalize(input, dim=-1)
        return self.dropout(self.fc(input))
