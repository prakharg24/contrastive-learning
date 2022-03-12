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

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def triplet_contrastive_loss(x, B_pos, B_neg):
    loss = -torch.mean(torch.matmul(B_pos, x)) + torch.mean(torch.matmul(B_neg, x))
    return loss




class SelfconLoss(nn.Module):
    def __init__(self, lam):
        super(SelfconLoss, self).__init__()
        self.lam = lam

    def regularization_loss(self, model):
        return self.lam / 2 * torch.linalg.matrix_norm(torch.square(torch.matmul(model.linear.weight, model.linear.weight.T)), ord='fro')

    def forward(self, z_i, z_j, model):
        loss = 0
        n = len(z_i)
        for index, (d_i, d_j) in enumerate(zip(z_i, z_j)):
            loss += 2 * torch.dot(d_i, d_j)
            mask = torch.ones(n).double().cuda()
            mask[index] = 0

            loss -= torch.dot(torch.matmul(z_i, d_i), mask)/(2 * n - 2)
            loss -= torch.dot(torch.matmul(z_i, d_j), mask)/(2 * n - 2)
            loss -= torch.dot(torch.matmul(z_j, d_i), mask)/(2 * n - 2)
            loss -= torch.dot(torch.matmul(z_j, d_j), mask)/(2 * n - 2)
        loss = -loss / (2 * n)
        loss += self.regularization_loss(model)
        return loss


class DataHandler(Dataset):
    def generate_random_mask(self, size):
        random_mask = np.diag(np.rint(np.random.rand(size)))
        return torch.tensor(random_mask)

    def __init__(self, X):
        self.X = X
        self.random_mask = self.generate_random_mask(X.size()[1])

    def __getitem__(self, index):
        x = self.X[index]
        x1 = torch.matmul(self.random_mask, x)
        x2 = torch.matmul((1 - self.random_mask), x)
        return x1, x2

    def __len__(self):
        return len(self.X)

def train(train_loader, model, criterion, optimizer, loss_fn):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)
        if loss_fn == "NTXENT":
            loss = criterion(z_i, z_j)
        else:
            loss = criterion(z_i, z_j, model)
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

def contrastive_training(r, d, x, loss_fn, batch_size, num_epochs, lr, lam, cuda=True):
    # Data Loader
    x = torch.tensor(x)
    # print(f"Train Data Shape: {x.size()}")
    train_loader = DataLoader(DataHandler(x), shuffle=True, batch_size=batch_size, drop_last=True)
    # Model
    model = linear_CL_Model(d, r).double().cuda()
    # print(f"Model Size: {model.linear.weight.size()}")
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn == "NTXENT":
        criterion = NT_Xent(batch_size, 0.5, 1)
    else:
        criterion = SelfconLoss(lam)

    epoch_iterator = tqdm(range(num_epochs), desc='Epochs', position=0)
    loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in epoch_iterator:
        loss_epoch = train(train_loader, model, criterion, optimizer, loss_fn)
        loss_log.set_description_str(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss_epoch / len(train_loader)}")
    return model
