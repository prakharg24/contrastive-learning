import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import torch.nn as nn
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
from test_utils import sinedistance_eigenvectors
import torch
import torch.nn as nn
import torch.nn.functional as F


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # numerator = torch.exp(positive_samples).squeeze()
        # denominator = torch.sum(torch.exp(negative_samples), dim=1) + numerator
        # loss = -torch.log((numerator / denominator) + 1e-10).mean()

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def triplet_contrastive_loss(x, B_pos, B_neg):
    loss = -torch.mean(torch.matmul(B_pos, x)) + torch.mean(torch.matmul(B_neg, x))
    return loss


def regularization_loss(lam, model):
    return lam / 2 * torch.linalg.matrix_norm(torch.square(torch.matmul(model.linear.weight, model.linear.weight.T)), ord='fro')


class DataHandler(Dataset):
    def generate_random_mask(self, size, mask_percentage):
        mask_size = int(mask_percentage*size)
        random_mask = np.concatenate([np.ones(size - mask_size), np.zeros(mask_size)])
        np.random.shuffle(random_mask)

        random_mask = np.diag(random_mask)
        return torch.tensor(random_mask)

    def __init__(self, X, mask_percentage, flip_mask=False, same_mask=True):
        self.X = X
        self.input_size = X.size()[1]
        self.flip_mask = flip_mask
        self.same_mask = same_mask
        self.mask_percentage = mask_percentage
        self.mask1 = self.generate_random_mask(self.input_size, mask_percentage)
        if flip_mask:
            self.mask2 = torch.diag(1 - torch.diag(self.mask1))
        else:
            self.mask2 = self.generate_random_mask(self.input_size, mask_percentage)

    def __getitem__(self, index):
        x = self.X[index]

        # mask1 = self.generate_random_mask(self.input_size)
        if self.same_mask:
            mask1 = self.mask1
            mask2 = self.mask2
        else:
            mask1 = self.generate_random_mask(self.input_size, self.mask_percentage)
            if self.flip_mask:
                mask2 = torch.diag(1 - torch.diag(self.mask1))
            else:
                mask2 = self.generate_random_mask(self.input_size, self.mask_percentage)
        x1 = torch.matmul(mask1, x)
        x2 = torch.matmul(mask2, x)
        return x1, x2

    def __len__(self):
        return len(self.X)

def train(train_loader, model, criterion, optimizer, lam, device, single_layer):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)
        loss = criterion(z_i, z_j)
        if single_layer:
            loss += regularization_loss(lam, model)
        loss.backward()

        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch


class linear_CL_Model(torch.nn.Module):
    def __init__(self, d, r):
        super(linear_CL_Model, self).__init__()
        self.linear = torch.nn.Linear(d, r, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

class double_CL_Model(torch.nn.Module):
    def __init__(self, d, r):
        super(double_CL_Model, self).__init__()
        self.linear_1 = torch.nn.Linear(d, int((r+d)/2), bias=True)
        self.linear_2 = torch.nn.Linear(int((r+d)/2), r, bias=True)

    def forward(self, x):
        out = self.linear_2(F.relu(self.linear_1(x)))
        return out
    def get_latent_representation(self, x):
        return self.forward(x)

def contrastive_training(r, d, x, ustar, loss_fn, batch_size, num_epochs, single_layer, lr, lam, patience, mask_percentage, cuda=True, flip=True, fix=True):
    # Data Loader
    x = torch.tensor(x)
    device = 'cuda' if cuda else 'cpu'
    # print(f"Train Data Shape: {x.size()}")
    train_loader = DataLoader(DataHandler(x, mask_percentage, flip_mask=flip, same_mask=fix), shuffle=True, batch_size=batch_size, drop_last=True)
    # Model
    if single_layer:
        model = linear_CL_Model(d, r).double().to(device)
    else:
        model = double_CL_Model(d, r).double().to(device)
    # print(f"Model Size: {model.linear.weight.size()}")
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NT_Xent(batch_size, 0.5, 1)

    epoch_iterator = tqdm(range(num_epochs), desc='Epochs')
    # loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    patience_steps = 0
    min_loss = float("inf")
    best_model = model.state_dict()
    for epoch in epoch_iterator:
        loss_epoch = train(train_loader, model, criterion, optimizer, lam, device, single_layer)
        if loss_epoch < min_loss:
            min_loss = loss_epoch
            patience_steps = 0
            best_model = model.state_dict()
        else:
            patience_steps += 1
        if patience_steps > patience:
            model.load_state_dict(best_model)
            break
        epoch_iterator.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss_epoch / len(train_loader)}")
    return model
