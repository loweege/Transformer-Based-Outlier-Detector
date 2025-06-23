from torch.utils.data import Dataset
import torch

class SequenceDataset(Dataset):
    """
    Dataset that generates sliding windows of fixed length (e.g., 30),
    splits each window into k parts, and for each part, uses the first (n-1)
    embeddings as input and the last as the target.
    """
    def __init__(self, embeddings: torch.Tensor, window_size: int = 30, splits: int = 3):
        assert window_size % splits == 0, "Window size must be divisible by number of splits"
        self.embeds = embeddings
        self.window_size = window_size
        self.split_size = window_size // splits
        self.splits = splits
        self.num_windows = embeddings.size(0) - window_size + 1

    def __len__(self):
        return self.num_windows * self.splits

    def __getitem__(self, idx):
        window_idx = idx // self.splits
        split_idx = idx % self.splits
        window = self.embeds[window_idx : window_idx + self.window_size]
        
        part = window[split_idx * self.split_size : (split_idx + 1) * self.split_size]
        x = part[:-1]
        y = part[-1]
        return x, y