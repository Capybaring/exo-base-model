"""Train-only pattern clustering. No learned token coordinates are clustered."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .model import pattern_windows, z_normalize


def fit_pattern_bank(records, train_indices, collator, target_dim, context=4,
                     n_patterns=16, max_windows=20000, seed=1):
    from tslearn.clustering import KShape
    if min(context, n_patterns, max_windows) < 1:
        raise ValueError('Pattern counts and window size must be positive')
    rng = np.random.default_rng(seed)
    reservoir = np.empty((max_windows, context), dtype=np.float32)
    seen = 0
    for batch in DataLoader(Subset(records, train_indices), batch_size=1, collate_fn=collator):
        windows, complete = pattern_windows(batch, target_dim, context)
        candidates = windows[complete]
        candidates = candidates[candidates.std(-1, correction=0) > 1e-8]
        for row in z_normalize(candidates).numpy():
            if seen < max_windows:
                reservoir[seen] = row
            else:
                slot = int(rng.integers(seen+1))
                if slot < max_windows:
                    reservoir[slot] = row
            seen += 1
    if seen == 0:
        raise ValueError('No complete, nonconstant training X pattern windows; reduce delay_context or use variant=exo')
    candidates = reservoir[:min(seen, max_windows)]
    unique = np.unique(np.round(candidates, 6), axis=0)
    count = min(n_patterns, len(unique))
    if count == 1:
        centers = unique[:1]
    else:
        estimator = KShape(n_clusters=count, max_iter=30, n_init=1, random_state=seed)
        estimator.fit(candidates[..., None])
        centers = estimator.cluster_centers_[..., 0].astype(np.float32)
    if not np.isfinite(centers).all():
        raise ValueError('Pattern clustering produced nonfinite centroids')
    return centers, {'method': 'kshape', 'descriptor': 'z-normalized fixed-duration X patch means',
        'scope': 'train records only (historical and declared-known future X)',
        'candidate_windows': seen, 'sampled_windows': len(candidates), 'patterns': len(centers),
        'train_record_ids': [int(records.records[i][0]) for i in train_indices]}
