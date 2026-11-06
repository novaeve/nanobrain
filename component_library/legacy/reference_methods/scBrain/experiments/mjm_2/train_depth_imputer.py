"""
Train a lightweight DepthImputer: X[B,140] -> depth_norm[B,1]
Uses only cells with has_depth=True (~19.5%, ~368K cells).
Saves: data/depth_imputer.pth
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

NPZ_PATH = './data/global_data_v2.npz'
OUT_PTH  = './data/depth_imputer.pth'
DEVICE   = 'cuda:0'
BS       = 2048
EPOCHS   = 100
LR       = 1e-3


class DepthImputer(nn.Module):
    def __init__(self, input_dim=140, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print('Loading data...')
    data = np.load(NPZ_PATH, allow_pickle=True)
    X         = torch.from_numpy(data['X']).float()
    depth     = torch.from_numpy(data['depth_norm']).float()
    has_depth = torch.from_numpy(data['has_depth']).bool()

    X_norm = torch.log1p(X)

    # Only train on cells with real depth
    X_d   = X_norm[has_depth]
    dep_d = depth[has_depth].unsqueeze(-1)
    print(f'Training DepthImputer on {len(X_d):,} cells...')

    dataset = TensorDataset(X_d, dep_d)
    loader  = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=True)

    model     = DepthImputer(input_dim=140, hidden=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, db in loader:
            xb, db = xb.to(DEVICE), db.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, db)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), OUT_PTH)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | MSE={avg:.6f} | best={best_loss:.6f}')

    print(f'Done. Best depth imputer saved to {OUT_PTH}')


if __name__ == '__main__':
    main()
