import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import math
import numpy as np
from tqdm import tqdm

from contrastive_learning import DataHandler
import torch.nn.functional as F

'''
POTENTIAL CHANGES FOR EXPERIMENTS:

1. Non-linear Auto-Encoder [Change activation to non-linear, will it make any change?]
2. Change in loss function [TODO after theoretical considerations. Currently using MSE Loss]
3. Learn noise [TODO how? Is it a part of relaxing loss functions?]
4. How about latent representation comparisons? [Implemented retrieval]
'''

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
            self.encoder_1 = nn.Linear(d, int((r+d)/2), bias=True)
            self.encoder_2 = nn.Linear(int((r+d)/2), r, bias=True)
            self.decoder_1 = nn.Linear(r, int((r+d)/2), bias=True)
            self.decoder_2 = nn.Linear(int((r+d)/2), d, bias=True)


    def forward(self, x):
        if self.single_layer:
            latent_rep = self.encoder(x)
            prediction = self.decoder(latent_rep)
        else:
            latent_rep = F.relu(self.encoder_2(F.relu(self.encoder_1(x))))
            prediction = self.decoder_2(F.relu(self.decoder_1(latent_rep)))
        return prediction

    def get_latent_representation(self, x):
        if self.single_layer:
            latent_rep = self.encoder(x)
        else:
            latent_rep = F.relu(self.encoder_2(F.relu(self.encoder_1(x))))
        return latent_rep

def sequential_linear_block(in_layers, out_layers, requires_relu=False):
    if requires_relu:
        return nn.Sequential(nn.Linear(in_layers, out_layers, bias=False),
                        nn.ReLU())
    else:
        return nn.Linear(in_layers, out_layers, bias=False)


def regularization_loss(weight, lam):
    return lam / 2 * torch.linalg.matrix_norm(torch.square(torch.matmul(weight, weight.T)), ord='fro')

def auto_encoder(d, r, X, batch_size, num_epochs, lr, single_layer, requires_relu, lam, patience, mask_percentage, cuda=True):
    X = torch.tensor(X)
    device = 'cuda' if cuda else 'cpu'

    # Load training data
    train_dataloader = DataLoader(DataHandler(X, mask_percentage, flip_mask=False), batch_size=batch_size, shuffle=True, drop_last=True)
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
            if single_layer:
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
