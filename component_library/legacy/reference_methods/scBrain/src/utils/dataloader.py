import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SpatialSEAAD_Dataset(Dataset):
    def __init__(self, X, y_class, y_subclass, y_supertype, batch_donor, spatial_coords, cps, confidence, meta_info):
        self.X = torch.from_numpy(X).float()
        self.y_class = torch.from_numpy(y_class).long()
        self.y_subclass = torch.from_numpy(y_subclass).long()
        self.y_supertype = torch.from_numpy(y_supertype).long()
        self.batch_donor = torch.from_numpy(batch_donor).long()
        self.spatial_coords = torch.from_numpy(spatial_coords).float()
        self.cps = torch.from_numpy(cps).float()
        self.confidence = torch.from_numpy(confidence).float()

        self.num_c = meta_info.item().get('num_class')
        self.num_sc = meta_info.item().get('num_subclass')
        self.num_st = meta_info.item().get('num_supertype')

        # Per-donor z-score normalized spatial coords (scaled to ~gene expression range)
        self.spatial_zscore = self._zscore_spatial_per_donor(
            spatial_coords, batch_donor, scale=2.0)

    @staticmethod
    def _zscore_spatial_per_donor(spatial, batch_donor, scale=2.0):
        """Z-score normalize spatial coords per donor, then scale."""
        spatial_norm = np.zeros_like(spatial, dtype=np.float32)
        for d in np.unique(batch_donor):
            mask = batch_donor == d
            s = spatial[mask]
            mean = s.mean(axis=0)
            std = s.std(axis=0) + 1e-8
            spatial_norm[mask] = (s - mean) / std * scale
        return torch.from_numpy(spatial_norm).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'spatial': self.spatial_coords[idx],
            'spatial_zscore': self.spatial_zscore[idx],
            'batch_id': self.batch_donor[idx],
            'y_class': self.y_class[idx],
            'y_subclass': self.y_subclass[idx],
            'y_supertype': self.y_supertype[idx],
            'cps': self.cps[idx],
            'confidence': self.confidence[idx]
        }

def build_dataloaders(npz_path, batch_size=256, seed=42, num_workers=8):
    data = np.load(npz_path, allow_pickle=True)
    
    # 提取所有特征
    X_all, y_c_all, y_sc_all, y_st_all = data['X'], data['y_class'], data['y_subclass'], data['y_supertype']
    batch_all, spatial_all = data['batch_donor'], data['spatial']
    cps_all, confidence_all = data['cps'], data['y_supertype_confidence']
    meta_info = data['meta']
    
    unique_donors = np.unique(batch_all)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)
    
    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]  
    
    mask_train = np.isin(batch_all, train_donors)
    mask_val   = np.isin(batch_all, val_donors)
    mask_test  = np.isin(batch_all, test_donors)
    
    train_dataset = SpatialSEAAD_Dataset(
        X_all[mask_train], y_c_all[mask_train], y_sc_all[mask_train], y_st_all[mask_train], 
        batch_all[mask_train], spatial_all[mask_train], cps_all[mask_train], confidence_all[mask_train], meta_info
    )
    val_dataset = SpatialSEAAD_Dataset(
        X_all[mask_val], y_c_all[mask_val], y_sc_all[mask_val], y_st_all[mask_val], 
        batch_all[mask_val], spatial_all[mask_val], cps_all[mask_val], confidence_all[mask_val], meta_info
    )
    test_dataset = SpatialSEAAD_Dataset(
        X_all[mask_test], y_c_all[mask_test], y_sc_all[mask_test], y_st_all[mask_test], 
        batch_all[mask_test], spatial_all[mask_test], cps_all[mask_test], confidence_all[mask_test], meta_info
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader