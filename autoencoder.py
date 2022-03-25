import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import math
import numpy as np
from tqdm import tqdm

from contrastive_learning import DataHandler


class AutoEncoder(nn.Module):
    def __init__(self, d, r, single_layer, requires_relu):
        super(AutoEncoder, self).__init__()

        self.input_dim = d
        self.latent_dim = r
        self.requires_relu = requires_relu
        self.single_layer = single_layer

        if(self.single_layer):
            self.encoder = nn.Linear(d, r, bias=False)
            self.decoder = nn.Linear(r, d, bias=False)
        else:
            layer_dim = [self.input_dim]
            hidden_dim = 2**math.ceil(math.log(self.input_dim, 2))
            layer_dim.append(hidden_dim)
            while hidden_dim > self.latent_dim:
                hidden_dim = int(max(self.latent_dim, hidden_dim/2))
                layer_dim.append(hidden_dim)

            num_layers = len(layer_dim)
            encoder_layers = [sequential_linear_block(in_layers, out_layers, requires_relu) for in_layers, out_layers in zip(layer_dim[:num_layers-1], layer_dim[1:num_layers-1])]
            self.encoder = nn.Sequential(*encoder_layers, nn.Linear(2**math.ceil(math.log(self.latent_dim, 2)), self.latent_dim))

            layer_dim.reverse()
            decoder_layers = [sequential_linear_block(in_layers, out_layers, requires_relu) for in_layers, out_layers in zip(layer_dim[:num_layers-1], layer_dim[1:num_layers-1])]
            self.decoder = nn.Sequential(*decoder_layers, nn.Linear(2**math.ceil(math.log(self.input_dim, 2)), self.input_dim))

    def forward(self, x):
        latent_rep = self.encoder(x)
        prediction = self.decoder(latent_rep)

        return prediction

    def get_latent_representation(self, x):
        latent_rep = self.encoder(x)

        return latent_rep

def sequential_linear_block(in_layers, out_layers, requires_relu=False):
    if requires_relu:
        return nn.Sequential(nn.Linear(in_layers, out_layers, bias=False),
                        nn.ReLU())
    else:
        return nn.Linear(in_layers, out_layers, bias=False)


def regularization_loss(weight, lam):
    return lam / 2 * torch.linalg.matrix_norm(torch.square(torch.matmul(weight, weight.T)), ord='fro')

def auto_encoder(d, r, X, batch_size, num_epochs, lr, single_layer, requires_relu, lam, patience, mask_percentage=0.5, cuda=True):
    X = torch.tensor(X)
    device = 'cuda' if cuda else 'cpu'

    # Load training data
    train_dataloader = DataLoader(DataHandler(X, flip_mask=True, mask_percentage=mask_percentage), batch_size=batch_size, shuffle=True, drop_last=True)
    # Model
    model = AutoEncoder(d, r, single_layer, requires_relu)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    model.double().to(device)

    loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    patience_steps = 0
    min_loss = float("inf")
    best_model = model.state_dict()
    for epoch in tqdm(range(num_epochs)):
        loss_epoch = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_data = torch.cat(batch_data, 0)
            batch_data = batch_data.to(device)
            prediction = model(batch_data)
            loss = criterion(prediction, batch_data)
            loss += regularization_loss(model.encoder.weight, lam)
            loss.backward()
            optimizer.step()
            loss_log.set_description_str(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss / len(train_dataloader)}")
            loss_epoch += loss.item()
        if loss_epoch < min_loss:
            min_loss = loss_epoch
            patience_steps = 0
            best_model = model.state_dict()
        else:
            patience_steps += 1
        if patience_steps > patience:
            model.load_state_dict(best_model)
            break
    return model
