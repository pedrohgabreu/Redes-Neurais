import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.0):
        super(FlexibleMLP, self).__init__()
        
        layers = []
        in_features = input_dim
        
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = units
            
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)