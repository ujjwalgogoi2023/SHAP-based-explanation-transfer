"""
PyTorch Dataset wrapping harmonised flow-feature arrays.

Each flow is treated as a single timestep sequence of length 1 by default
(sequence_length=1 in config), i.e. the LSTM sees one CNN-extracted feature
vector per flow. Set sequence_length > 1 to group consecutive flows (e.g.
same-session windows) into a real sequence before feeding the LSTM.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 1):
        assert sequence_length >= 1
        self.sequence_length = sequence_length
        self.n_features = X.shape[1]

        if sequence_length == 1:
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
        else:
            n_windows = len(X) - sequence_length + 1
            if n_windows <= 0:
                raise ValueError("sequence_length longer than dataset")
            self.X = np.stack([X[i:i + sequence_length] for i in range(n_windows)]).astype(np.float32)
            self.y = y[sequence_length - 1:].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.sequence_length == 1:
            x = x.reshape(1, self.n_features)  # (seq_len=1, n_features)
        return torch.from_numpy(x), torch.tensor(self.y[idx])


def make_dataloaders(splits, sequence_length: int, batch_size: int, num_workers: int = 0):
    from torch.utils.data import DataLoader

    train_ds = FlowDataset(splits["X_train"], splits["y_train"], sequence_length)
    val_ds = FlowDataset(splits["X_val"], splits["y_val"], sequence_length)
    test_ds = FlowDataset(splits["X_test"], splits["y_test"], sequence_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
