import numpy as np
import torch
from torch import nn

class GRUModel(nn.Module):

    def __init__(self, feature_count):
        super().__init__()
        self.rnn = nn.GRU(feature_count, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, inputs):
        return self.head(self.rnn(inputs)[0]).squeeze(-1)

def fit_gru(features, target, capacity, sequences, seed=42):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = sequences.ravel()
    mean = np.nanmean(features[rows], axis=0)
    std = np.nanstd(features[rows], axis=0)
    std = np.where(std > 1e-06, std, 1)
    scaled = scale_features(features, mean, std)
    model = GRUModel(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    for epoch in range(8):
        model.train()
        order = rng.permutation(len(sequences))
        loss_sum = 0
        for start in range(0, len(order), 256):
            batch = sequences[order[start:start + 256]]
            inputs = torch.from_numpy(scaled[batch])
            labels = torch.from_numpy((target[batch] / capacity[batch]).astype('float32'))
            optimizer.zero_grad()
            loss = ((model(inputs) - labels) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_sum += loss.item()
        print('GRU epoch', epoch + 1, 'loss sum', loss_sum, flush=True)
    return (model, mean, std)

def scale_features(features, mean, std):
    return np.nan_to_num((features - mean) / std, nan=0, posinf=0, neginf=0).clip(-20, 20).astype('float32')
