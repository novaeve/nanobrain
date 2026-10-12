"""Frozen train-only annotation targets, distinct missing and unseen masks."""
from pathlib import Path
import sys
import pandas as pd
import torch


class AnnotationTargets:
    def __init__(self, benchmark_dir):
        self.vocabulary=pd.read_parquet(Path(benchmark_dir)/'annotation_vocab_train.parquet')
        self.columns=['annotation_label','annotation_label_source','annotation_label_level']
        keys=list(self.vocabulary[self.columns].itertuples(index=False,name=None))
        if len(set(keys))!=len(keys):raise ValueError('duplicate train label keys')
        self.lookup={key:i for i,key in enumerate(keys)}

    def encode(self, cells, device='cpu'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pretrain_dataset'))
        from brainwm_data.task_labels import canonicalize_annotation_frame
        frame=canonicalize_annotation_frame(cells)
        available=frame.annotation_available.to_numpy(dtype=bool)
        ids=torch.tensor([self.lookup.get(key,-1) if valid else -1 for key,valid in
            zip(frame[self.columns].itertuples(index=False,name=None),available)],device=device)
        known=ids>=0
        present=torch.tensor(available,device=device)
        return dict(annotation_target=ids,annotation_supervised=known,
                    annotation_missing=~present,annotation_unseen=present & ~known)


def annotation_loss(logits, targets):
    keep=targets['annotation_supervised']
    if not keep.any():return logits.sum()*0
    return torch.nn.functional.cross_entropy(logits[keep],targets['annotation_target'][keep])
