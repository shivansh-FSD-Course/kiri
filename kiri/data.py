"""kiri/data.py"""
import numpy as np


class Dataset:
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *arrays):
        assert len({len(a) for a in arrays}) == 1
        self.arrays = arrays
    def __len__(self): return len(self.arrays[0])
    def __getitem__(self, idx): return tuple(a[idx] for a in self.arrays)


class DataLoader:
    """
    Iterates a dataset in mini-batches.

    Usage:
        loader = kiri.DataLoader((X, y), batch_size=64, shuffle=True)
        for X_batch, y_batch in loader:
            ...
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False):
        if isinstance(dataset, (tuple, list)):
            dataset = TensorDataset(*dataset)
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start+self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            items = [self.dataset[i] for i in batch]
            if isinstance(items[0], tuple):
                yield tuple(np.stack([item[k] for item in items]) for k in range(len(items[0])))
            else:
                yield np.stack(items)

    def __repr__(self):
        return f"DataLoader(n={len(self.dataset)}, batch_size={self.batch_size}, shuffle={self.shuffle})"
