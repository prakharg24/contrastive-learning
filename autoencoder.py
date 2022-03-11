import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import math
import numpy as np
from tqdm import tqdm

'''
POTENTIAL CHANGES FOR EXPERIMENTS:

1. Non-linear Auto-Encoder [Change activation to non-linear, will it make any change?]
2. Change in loss function [TODO after theoretical considerations. Currently using MSE Loss]
3. Learn noise [TODO how? Is it a part of relaxing loss functions?]
4. How about latent representation comparisons? [Implemented retrieval]
'''

class AutoEncoder(nn.Module):
    def __init__(self, d, r):
        super(AutoEncoder, self).__init__()
        
        self.input_dim = d 
        self.latent_dim = r

        layer_dim = [self.input_dim]
        hidden_dim = 2**math.ceil(math.log(self.input_dim, 2))
        layer_dim.append(hidden_dim)
        while hidden_dim > self.latent_dim:
            hidden_dim = int(max(self.latent_dim, hidden_dim/2))
            layer_dim.append(hidden_dim)

        num_layers = len(layer_dim)
        encoder_layers = [sequential_linear_block(in_layers, out_layers) for in_layers, out_layers in zip(layer_dim[:num_layers-1], layer_dim[1:num_layers-1])]
        self.encoder = nn.Sequential(*encoder_layers, nn.Linear(2**math.ceil(math.log(self.latent_dim, 2)), self.latent_dim))

        layer_dim.reverse()
        decoder_layers = [sequential_linear_block(in_layers, out_layers) for in_layers, out_layers in zip(layer_dim[:num_layers-1], layer_dim[1:num_layers-1])]
        self.decoder = nn.Sequential(*decoder_layers, nn.Linear(2**math.ceil(math.log(self.input_dim, 2)), self.input_dim))

    def forward(self, x):
        latent_rep = self.encoder(x)
        prediction = self.decoder(latent_rep)

        return prediction

    def get_latent_representation(self, x):
        latent_rep = self.encoder(x)

        return latent_rep

def sequential_linear_block(in_layers, out_layers):
    return nn.Sequential(nn.Linear(in_layers, out_layers),
                        nn.ReLU())


def auto_encoder(d, r, X, batch_size, num_epochs, lr):
    X = torch.Tensor(X)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load training data
    train_dataloader = DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # Model
    model = AutoEncoder(d, r)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    model.to(device)

    loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(num_epochs)):
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            prediction = model(batch_data)
            loss = criterion(prediction, batch_data)
            loss.backward()
            optimizer.step()
            loss_log.set_description_str(f"Epoch [{epoch}/{num_epochs}] Loss: {loss / len(train_dataloader)}")

    return model