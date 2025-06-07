import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

csv_path = 'datasets/SODIndoorLoc-main/SYL/Training_SYL_All_30.csv'
df = pd.read_csv(csv_path)

def extract_mac_and_sample_time(df):
    mac_columns = [col for col in df.columns if "MAC" in col]
    mac_vectors = df[mac_columns].values
    sample_times = df['SampleTimes'].values

    mac_tensor = torch.tensor(mac_vectors, dtype=torch.float32)
    ts_tensor = torch.tensor(sample_times, dtype=torch.float32).squeeze()
    return mac_tensor, ts_tensor

mac_tensor, ts_tensor = extract_mac_and_sample_time(df)

def extract_embeddings(mac_tensor):
    """
    Extract embeddings from the MAC address tensor using PCA.
    """
    pca = PCA(n_components=128)
    mac_tensor = pca.fit_transform(mac_tensor)
    return torch.tensor(mac_tensor, dtype=torch.float32)

embeds = extract_embeddings(mac_tensor)

'''
train a transformer with embeddings so that it is an autoregressive model
and it predicts the next signal embedding based on the previous ones
'''

# DO some data augmentation

class SequenceDataset(Dataset):
    """
    Dataset that generates sliding windows of fixed length (e.g., 30),
    using the first (seq_len - 1) embeddings as input and the last as target.
    """
    def __init__(self, embeddings: torch.Tensor, window_size: int = 30):
        self.embeds = embeddings
        self.window_size = window_size
        self.num_windows = embeddings.size(0) - window_size + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        window = self.embeds[idx : idx + self.window_size]
        x = window[:-1]
        y = window[-1]
        return x, y

class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class AutoregressiveTransformer(nn.Module):
    """
    Simple autoregressive Transformer model to predict the next embedding.
    """
    def __init__(self, 
                 embed_dim: int = 128, 
                 nhead: int = 8, 
                 num_layers: int = 3,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, src: torch.Tensor):
        src = src.transpose(0, 1)
        src = self.positional_encoding(src)
        encoded = self.transformer_encoder(src)
        last = encoded[-1] 
        return self.fc_out(last)


if __name__ == "__main__":
    window_size = 30
    seq_len = window_size - 1
    batch_size = 32
    lr = 1e-4
    epochs = 100

    dataset = SequenceDataset(embeds, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoregressiveTransformer(embed_dim=embeds.size(1))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            # y_batch shape: (batch, embed_dim)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "autoregressive_transformer.pth")






# divide the data so that you know wchic comes first and which comes second using the timestamp attribute

# divide the data into training and test sets, remember that this is time series data, so you need to take care of the order

# build the dataloaders for training and testing

# create the embeddings using the convolutional block defined above

#use convolutional block to learn the embeddings

'''
TO DO:

the embeddings needs to be done with a CNN like your thesis

create the signal embeddings, remember these are time series data, use
sk-learn to check if you can split the data automatically for time series

'''

#JUST DO THE EMBEDDINGS AND DIVIDE THEM IN ORDER, THEN MERGE THEM TOGETHER (KINDA)
# AND FINALLY GIVE IT AS INPUTS IN THE TRANSFORMER, CHECK THE PADDING AND MASKING
# IF THEY HAVE TO BE OF THE SAME LENGTH OR NOT