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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


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
    def generate_random_mask(self, size, mask_percentage=0.5):
        mask_size = int(mask_percentage*size)
        random_mask = np.concatenate([np.ones(size - mask_size), np.zeros(mask_size)])
        np.random.shuffle(random_mask)

        random_mask = np.diag(random_mask)
        return torch.tensor(random_mask)

    def __init__(self, X, flip_mask=False, mask_percentage=0.5):
        self.X = X
        self.input_size = X.size()[1]
        self.flip_mask = flip_mask
        self.mask_percentage = mask_percentage
        self.mask1 = self.generate_random_mask(self.input_size, self.mask_percentage)
        if flip_mask:
            self.mask2 = torch.diag(1 - torch.diag(self.mask1))
        else:
            self.mask2 = self.generate_random_mask(self.input_size, self.mask_percentage)

    def __getitem__(self, index):
        x = self.X[index]

        # mask1 = self.generate_random_mask(self.input_size)
        x1 = torch.matmul(self.mask1, x)

        # if self.flip_mask:
        #     mask2 = 1 - mask1
        # else:
        #     mask2 = self.generate_random_mask(self.input_size)
        x2 = torch.matmul(self.mask2, x)
        return x1, x2

    def __len__(self):
        return len(self.X)

def train(train_loader, model, criterion, optimizer, loss_fn, device):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)
        if loss_fn == "NTXENT":
            loss = criterion(z_i, z_j)
        else:
            z_i = torch.unsqueeze(z_i, 1)
            z_j = torch.unsqueeze(z_j, 1)

            batch_views = torch.cat((z_i, z_j), 1)
            loss = criterion(batch_views)
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

def contrastive_training(r, d, x, ustar, loss_fn, batch_size, num_epochs, lr, lam, patience, cuda=True):
    # Data Loader
    x = torch.tensor(x)
    device = 'cuda' if cuda else 'cpu'
    # print(f"Train Data Shape: {x.size()}")
    train_loader = DataLoader(DataHandler(x, flip_mask=True), shuffle=True, batch_size=batch_size, drop_last=True)
    # Model
    model = linear_CL_Model(d, r).double().to(device)
    # print(f"Model Size: {model.linear.weight.size()}")
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn == "NTXENT":
        criterion = NT_Xent(batch_size, 0.5, 1)
    elif loss_fn == "SupConLoss":
        criterion = SupConLoss()

    epoch_iterator = tqdm(range(num_epochs), desc='Epochs')
    # loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    patience_steps = 0
    min_loss = float("inf")
    best_model = model.state_dict()
    for epoch in epoch_iterator:
        loss_epoch = train(train_loader, model, criterion, optimizer, loss_fn, device)
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
