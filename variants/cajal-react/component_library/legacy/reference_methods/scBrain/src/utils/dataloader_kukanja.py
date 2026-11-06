"""
Dataloader for Kukanja packed NPZ files (MS or EAE).
Supports N-level label hierarchies and generic metadata fields.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KukanjaDataset(Dataset):
    """Generic N-level spatial transcriptomics dataset."""

    def __init__(self, X, labels_list, label_names, sample_ids,
                 spatial, disease_score, spatial_norm=None):
        """
        Args:
            X:             [N, G] gene expression matrix
            labels_list:   list of N-len arrays, one per hierarchy level
            label_names:   list of str, e.g. ['level0','level1_5','level2','level3']
            sample_ids:    [N,] sample/donor IDs for splitting
            spatial:       [N, 2] spatial coordinates
            disease_score: [N,] disease-related continuous score
            spatial_norm:  [N, 2] normalized spatial coords (optional)
        """
        self.X = torch.from_numpy(X).float()
        self.labels = [torch.from_numpy(y).long() for y in labels_list]
        self.label_names = label_names
        self.sample_ids = torch.from_numpy(sample_ids).long()
        self.spatial = torch.from_numpy(spatial).float()
        self.disease_score = torch.from_numpy(disease_score).float()
        if spatial_norm is not None:
            self.spatial_norm = torch.from_numpy(spatial_norm).float()
        else:
            self.spatial_norm = self.spatial.clone()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {
            'X': self.X[idx],
            'spatial': self.spatial[idx],
            'spatial_norm': self.spatial_norm[idx],
            'batch_id': self.sample_ids[idx],
            'disease_score': self.disease_score[idx],
        }
        for i, name in enumerate(self.label_names):
            item[f'y_{name}'] = self.labels[i][idx]
        return item


def build_dataloaders_kukanja(npz_path, batch_size=512, seed=3028, num_workers=8):
    """Load Kukanja packed npz and build train/val/test dataloaders."""
    data = np.load(npz_path, allow_pickle=True)

    X = data['X']
    sample_ids = data['sample_ids']
    spatial = data['spatial']
    disease_score = data['disease_score']
    spatial_norm = data.get('spatial_norm', None)

    # Load label names from metadata
    meta = data['meta'].item()
    label_names = meta['label_names']  # e.g. ['level0','level1_5','level2','level3']
    n_levels = len(label_names)

    labels_list = [data[f'y_{name}'] for name in label_names]

    # Sample-based split
    unique_samples = np.unique(sample_ids)
    n_samples = len(unique_samples)
    np.random.seed(seed)
    np.random.shuffle(unique_samples)

    # Adaptive split ratios based on number of samples
    if n_samples <= 6:
        # Few samples: 2 test, 1 val, rest train
        n_test = 2
        n_val = 1
    elif n_samples <= 15:
        # Medium: 3 test, 2 val, rest train
        n_test = 3
        n_val = 2
    else:
        # Many: ~20% test, ~10% val, rest train
        n_test = max(2, int(n_samples * 0.2))
        n_val = max(1, int(n_samples * 0.1))

    test_samples = unique_samples[:n_test]
    val_samples = unique_samples[n_test:n_test + n_val]
    train_samples = unique_samples[n_test + n_val:]

    mask_train = np.isin(sample_ids, train_samples)
    mask_val = np.isin(sample_ids, val_samples)
    mask_test = np.isin(sample_ids, test_samples)

    def _make_ds(mask):
        sn = spatial_norm[mask] if spatial_norm is not None else None
        return KukanjaDataset(
            X[mask],
            [y[mask] for y in labels_list],
            label_names,
            sample_ids[mask],
            spatial[mask],
            disease_score[mask],
            sn,
        )

    train_ds = _make_ds(mask_train)
    val_ds = _make_ds(mask_val)
    test_ds = _make_ds(mask_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    info = {
        'label_names': label_names,
        'output_num': [meta[f'num_{name}'] for name in label_names],
        'n_genes': X.shape[1],
        'n_cells': X.shape[0],
        'n_train': mask_train.sum(),
        'n_val': mask_val.sum(),
        'n_test': mask_test.sum(),
        'train_samples': train_samples.tolist(),
        'val_samples': val_samples.tolist(),
        'test_samples': test_samples.tolist(),
    }
    return train_loader, val_loader, test_loader, info
