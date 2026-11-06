"""
Dataloader for global_data_v2.npz.
Returns all fields needed by mjm_2a/b/c trainers.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SpatialSEAAD_V2_Dataset(Dataset):
    def __init__(self, X, y_class, y_subclass, y_supertype, batch_donor,
                 spatial, spatial_tiled, cps, confidence,
                 depth_norm, has_depth,
                 cell_volume_norm, has_volume,
                 meta_info):
        self.X               = torch.from_numpy(X).float()
        self.y_class         = torch.from_numpy(y_class).long()
        self.y_subclass      = torch.from_numpy(y_subclass).long()
        self.y_supertype     = torch.from_numpy(y_supertype).long()
        self.batch_donor     = torch.from_numpy(batch_donor).long()
        self.spatial         = torch.from_numpy(spatial).float()
        self.spatial_tiled   = torch.from_numpy(spatial_tiled).float()
        self.cps             = torch.from_numpy(cps).float()
        self.confidence      = torch.from_numpy(confidence).float()
        self.depth_norm      = torch.from_numpy(depth_norm).float()
        self.has_depth       = torch.from_numpy(has_depth.astype(np.float32)).float()
        self.cell_volume_norm = torch.from_numpy(cell_volume_norm).float()
        self.has_volume      = torch.from_numpy(has_volume.astype(np.float32)).float()

        meta = meta_info.item()
        self.num_c  = meta.get('num_class')
        self.num_sc = meta.get('num_subclass')
        self.num_st = meta.get('num_supertype')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            'X':               self.X[idx],
            'spatial':         self.spatial[idx],
            'spatial_tiled':   self.spatial_tiled[idx],
            'batch_id':        self.batch_donor[idx],
            'y_class':         self.y_class[idx],
            'y_subclass':      self.y_subclass[idx],
            'y_supertype':     self.y_supertype[idx],
            'cps':             self.cps[idx],
            'confidence':      self.confidence[idx],
            'depth_norm':      self.depth_norm[idx],
            'has_depth':       self.has_depth[idx],
            'cell_volume_norm': self.cell_volume_norm[idx],
            'has_volume':      self.has_volume[idx],
        }


def build_dataloaders_v2(npz_path, batch_size=512, seed=42, num_workers=8):
    data = np.load(npz_path, allow_pickle=True)

    X_all           = data['X']
    y_c_all         = data['y_class']
    y_sc_all        = data['y_subclass']
    y_st_all        = data['y_supertype']
    batch_all       = data['batch_donor']
    spatial_all     = data['spatial']
    spatial_tiled_all = data['spatial_tiled']
    cps_all         = data['cps']
    conf_all        = data['y_supertype_confidence']
    depth_all       = data['depth_norm']
    has_depth_all   = data['has_depth']
    vol_all         = data['cell_volume_norm']
    has_vol_all     = data['has_volume']
    meta_info       = data['meta']

    # per-donor split (same logic as v1)
    unique_donors = np.unique(batch_all)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)
    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]

    def _slice(mask):
        return SpatialSEAAD_V2_Dataset(
            X_all[mask], y_c_all[mask], y_sc_all[mask], y_st_all[mask],
            batch_all[mask], spatial_all[mask], spatial_tiled_all[mask],
            cps_all[mask], conf_all[mask],
            depth_all[mask], has_depth_all[mask],
            vol_all[mask], has_vol_all[mask],
            meta_info,
        )

    mask_train = np.isin(batch_all, train_donors)
    mask_val   = np.isin(batch_all, val_donors)
    mask_test  = np.isin(batch_all, test_donors)

    train_ds = _slice(mask_train)
    val_ds   = _slice(mask_val)
    test_ds  = _slice(mask_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader
